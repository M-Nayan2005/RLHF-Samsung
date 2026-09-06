-- =============================================================
-- Tier 3 / Tier 4 owned tables. Separate schema namespace so it's
-- obvious these are NOT tier1/tier2 tables. Tier 3 code has
-- read/write access to `tier3` schema only.
--
-- Ownership: `processed_annotations` and `effort_population_stats` are
-- Dev 1's. `replay_buffer` and `ppo_training_runs` are Dev 2's. The whole
-- file ships together because a half-applied schema is worse than either
-- half, and because `docker-entrypoint-initdb.d` applies files atomically.
--
-- Idempotent throughout: every statement is IF NOT EXISTS / ON CONFLICT, so
-- re-applying against a live database is a no-op rather than an error.
-- =============================================================
CREATE SCHEMA IF NOT EXISTS tier3;

-- Sequence/idempotency ledger — prevents double-processing a webhook
-- delivery and detects out-of-order ANNOTATION_UPDATED events for the
-- same task (e.g. a correction-of-a-correction).
CREATE TABLE IF NOT EXISTS tier3.processed_annotations (
    annotation_id       TEXT PRIMARY KEY,
    task_id             TEXT NOT NULL,
    event_id            TEXT NOT NULL,          -- from RedisEventEnvelope
    sequence_no         BIGINT NOT NULL,        -- monotonic per task_id, derived from enqueued_at
    processed_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    accepted            BOOLEAN NOT NULL,
    reject_reason       TEXT
);
CREATE INDEX IF NOT EXISTS idx_processed_annotations_task ON tier3.processed_annotations(task_id, sequence_no);

-- Rolling population stats for Z-score normalization of Delta_E.
-- Updated incrementally (Welford's algorithm) by the worker, one row.
CREATE TABLE IF NOT EXISTS tier3.effort_population_stats (
    id            SMALLINT PRIMARY KEY DEFAULT 1,
    count         BIGINT NOT NULL DEFAULT 0,
    mean          DOUBLE PRECISION NOT NULL DEFAULT 0,
    m2            DOUBLE PRECISION NOT NULL DEFAULT 0,   -- Welford running sum of squares
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (id = 1)
);
INSERT INTO tier3.effort_population_stats (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ---------------------------------------------------------------
-- The staged experience tuples waiting for Tier 4.
--
-- NAMING (Dev 2): `docs/authority.md` C1 rejects the "Offline Replay Buffer"
-- by name, because PPO is strictly on-policy and training on tuples collected
-- under superseded weights breaks the importance-sampling ratio. What this
-- table actually implements is the ratified streaming-rollout design — one
-- `model_version` per batch, consume-once, no random resampling over history.
-- The behaviour is correct; only the name is inherited. Left as `replay_buffer`
-- so it matches the Tier 3/4 plan and the Pydantic contract, and recorded here
-- so nobody concludes C1 was reopened.
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tier3.replay_buffer (
    tuple_id            UUID PRIMARY KEY,
    wiggle_seed         TEXT NOT NULL,
    task_id             TEXT NOT NULL,
    -- UNIQUE is the second line of defence on idempotency. Dev 1's sequence
    -- ledger is the first, but the two run in separate transactions, so a
    -- redelivery racing two workers can clear the ledger check twice. The
    -- writer relies on this constraint via ON CONFLICT DO NOTHING.
    annotation_id       TEXT NOT NULL UNIQUE,

    state_s_t           JSONB NOT NULL,          -- M_initial polygon
    action_a_t          JSONB NOT NULL,          -- M_wiggled polygon
    reward_r_t          DOUBLE PRECISION NOT NULL,

    delta_iou           DOUBLE PRECISION NOT NULL,
    delta_e_norm        DOUBLE PRECISION NOT NULL,
    alpha               DOUBLE PRECISION NOT NULL,
    beta                DOUBLE PRECISION NOT NULL,

    model_version       TEXT NOT NULL,           -- which LoRA checkpoint produced action_a_t
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),

    consumed_by_ppo     BOOLEAN NOT NULL DEFAULT FALSE,
    consumed_at         TIMESTAMPTZ,
    ppo_batch_id        UUID
);
-- Tier 4 pulls the oldest N unconsumed rows of a single model_version.
CREATE INDEX IF NOT EXISTS idx_replay_buffer_pull
    ON tier3.replay_buffer (model_version, consumed_by_ppo, created_at)
    WHERE consumed_by_ppo = FALSE;

-- Diversity-sampler support (plan §4). Nullable and additive: Tier 4 logs the
-- class distribution of each batch to catch the catastrophic-forgetting risk
-- flagged against batch-size-64. The label lives on the frozen Tier 1 schema,
-- which Tier 3 may not query, so it rides along on the WiggleCacheEntry
-- instead. NULL simply means the cache entry did not carry one.
ALTER TABLE tier3.replay_buffer ADD COLUMN IF NOT EXISTS label TEXT;
-- Honeypot tuples carry a real gold mask, so their delta_iou is the strict
-- two-term form rather than the production fallback. Tier 4 needs to be able
-- to tell them apart before mixing them in one batch (open question Q6).
ALTER TABLE tier3.replay_buffer ADD COLUMN IF NOT EXISTS is_honeypot BOOLEAN NOT NULL DEFAULT FALSE;

-- Tier 4: record of every PPO training run, for audit + rollback.
CREATE TABLE IF NOT EXISTS tier3.ppo_training_runs (
    batch_id            UUID PRIMARY KEY,
    model_version_in    TEXT NOT NULL,           -- checkpoint the batch was collected under (on-policy check)
    model_version_out   TEXT,                    -- new checkpoint produced, null until training completes
    batch_size          INT NOT NULL,
    tuple_ids           UUID[] NOT NULL,
    actor_loss          DOUBLE PRECISION,
    critic_loss         DOUBLE PRECISION,
    mean_reward         DOUBLE PRECISION,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'running',  -- running | succeeded | failed | discarded_stale_policy
    deployed_via        TEXT,                    -- e.g. 'blue_green:green-container-7'
    error_message       TEXT
);
