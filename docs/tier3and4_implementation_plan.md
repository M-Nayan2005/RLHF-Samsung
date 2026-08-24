# Tier 3 + Tier 4 Implementation Plan
## RLHF Reward Calculation, Replay Buffer & PPO/LoRA Training Loop

Team: 3 developers. Picks up exactly where Tier 2 left off — the boundary is
`telemetry:ingest` in Redis. Tier 1/2's Postgres tables and schema files are
**never touched** by this codebase.

---

## 0. Critical Dependency — Read This First

`LSAnnotationUpdatedPayload` (frozen, Tier 2-owned) carries the human's final
`result` and `effort_telemetry`, but **no mask geometry for `M_initial` or
`M_wiggled`**. Without those, `ΔIoU = IoU(M_final) − IoU(M_initial)` cannot be
computed from the webhook alone, and per the read-only boundary rule Tier 3
cannot query Tier 1/2's Postgres tables to fetch them either.

**Resolution**: Tier 2's `serving_ui`, at the moment it serves a wiggled mask
to Label Studio, must cache the pair under a Redis key namespaced
`wiggle_cache:{wiggle_seed}` (TTL ~24h) — see `WiggleCacheEntry` in
`common/schemas/tier3_rlhf.py`. This is a Redis write, not Postgres, and adds
a new key rather than modifying any frozen schema — it does not violate
either rule. **Get this confirmed/shipped by whoever owns `serving_ui` before
Dev 1 starts**, or the geometric half of the reward is unbuildable.

If that cache entry is missing or expired when a webhook arrives, the
consumer must not crash — see §2 error handling.

**Ground-truth caveat**: `IoU(M_initial)` / `IoU(M_final)` in the diagram
implicitly assumes comparison against a "true" mask, which doesn't exist for
non-honeypot tasks. In practice `Δ_IoU` is computed as `IoU(M_final,
M_initial)` — i.e. how much the human's correction diverged from the
AI's baseline — used as the accuracy-gain proxy, exactly as
`tier3.docx` describes: "ΔIoU measures the spatial divergence fixed by the
human." For honeypot tasks (where a real gold mask exists), a stricter
`IoU(M_final, ground_truth)` can be substituted — flagged as a v2 enhancement,
not required tonight.

---

## 1. New Contracts

All in `common/schemas/tier3_rlhf.py` (new file, does not touch frozen
modules): `WiggleCacheEntry`, `SequenceCheckResult`, `BiometricSignals`,
`EffortWeights`, `RawEffortScore`, `NormalizedEffortScore`, `GeometricDelta`,
`EDRDEReward`, `ExperienceTuple`, `RolloutBatchReadyEvent`.

Reward formula implemented exactly as specified:
```
Delta_E_raw  = w1*C + w2*L_path + w3*T_dwell
Delta_E_norm = (Delta_E_raw - population_mean) / population_stddev   # Z-score
Delta_IoU    = IoU(M_final, M_initial)
R_t          = alpha * Delta_IoU - beta * Delta_E_norm
```
Starting hyperparameters (tune later, not tonight): `alpha=1.0, beta=0.3`,
`w1=1.0, w2=0.01, w3=0.001` (chosen so clicks dominate, matching the original
doc's note that click-count was kept as the primary effort signal after
"Reward Formula Simplification").

---

## 2. Tier 3 Consumer Worker Architecture (Dev 1 + Dev 2)

Single logical pipeline, two owners split by node (see WBS). Runs as an
async Python process — **no synchronous drivers on the event loop**:

```
redis.asyncio  BRPOP telemetry:ingest
    -> deserialize RedisEventEnvelope (frozen schema, import don't redefine)
    -> [Dev 1] Sequence Check: look up tier3.processed_annotations by
       annotation_id (asyncpg). Duplicate -> ack + drop. Out-of-order
       (sequence_no for this task_id is lower than a previously seen one)
       -> ack + drop + log. New -> insert row, continue.
    -> [Dev 1] Fetch WiggleCacheEntry from Redis by wiggle_seed.
       Missing/expired -> log + drop (cannot compute geometry), do NOT crash
       the loop, do NOT block on retry — this is an acceptable data-loss
       edge case, log it loudly for ops visibility.
    -> [Dev 1] Extract BiometricSignals from effort_telemetry
    -> [Dev 1] Biometric Effort Engine: RawEffortScore
    -> [Dev 1] Z-Score Normalization: read/update
       tier3.effort_population_stats (Welford's online algorithm — single
       UPDATE, no read-then-write race condition), sanity filter drops
       tasks with implausible cursor velocity (path_length/dwell_time above
       a bot-like threshold) -> NormalizedEffortScore
    -> [Dev 2] Geometric Delta Engine: IoU(M_final from webhook result,
       M_initial from WiggleCacheEntry) -> GeometricDelta
    -> [Dev 2] E-DRDE Scalar Evaluator: combine GeometricDelta +
       NormalizedEffortScore -> EDRDEReward
    -> [Dev 2] State-Action-Reward Aggregator: build ExperienceTuple
       (state=M_initial, action=M_wiggled, reward=R_t, model_version=
       whatever tag was on the WiggleCacheEntry/served payload)
    -> [Dev 2] INSERT into tier3.replay_buffer (asyncpg, batched if
       throughput demands it)
```

Concurrency: run N worker coroutines pulling from the same Redis list (Redis
list pop is atomic, so this is safe fan-out) rather than one single-threaded
loop, to survive the "dozens of annotators submit simultaneously" spike
scenario from the original NFRs.

---

## 3. Offline Replay Buffer Schema (Dev 2)

`infra/sql/tier3_schema.sql` — new `tier3` Postgres schema (not `public`,
not touching `tier1_predictions`/`junior_queue`/etc):
- `tier3.processed_annotations` — idempotency/sequence ledger
- `tier3.effort_population_stats` — single-row rolling mean/stddev (Welford)
- `tier3.replay_buffer` — the experience tuples, indexed for Tier 4's pull
  pattern (`model_version, consumed_by_ppo, created_at`)
- `tier3.ppo_training_runs` — audit trail of every training run, model
  versions in/out, loss values, deployment target

---

## 4. Tier 4: Rollout Queue & PPO Trigger (Dev 3)

```
Tier 4 poller (asyncpg):
  SELECT tuple_id, ... FROM tier3.replay_buffer
  WHERE consumed_by_ppo = FALSE AND model_version = :current_serving_version
  ORDER BY created_at ASC LIMIT 64
  -- only fires the PPO step once this returns exactly 64 rows of the
  -- SAME model_version (on-policy purity: mixing tuples collected under
  -- two different checkpoints breaks the importance sampling ratio)

  IF count == 64:
    BEGIN transaction
      mark these 64 rows consumed_by_ppo=TRUE, ppo_batch_id=<new uuid>
    COMMIT
    -- flush semantics: rows are marked consumed, not deleted immediately,
    -- so a crashed training run can be diagnosed; a nightly cleanup job
    -- hard-deletes consumed rows older than e.g. 24h
    hand batch to PPO Actor-Critic step (Stable Baselines3 or custom loop)
    Critic Network: value estimate per state, computes advantage A_t
    Actor Network: policy gradient loss on the low-dim latent action space
    Backprop -> LoRA adapter weights only (frozen SAM2 backbone)
    Write tier3.ppo_training_runs row: losses, mean_reward, model_version_out
    -> Blue/Green deploy: load new LoRA weights into the idle container,
       swap Traefik/Nginx routing, flush the old container
    -> update "current_serving_version" pointer so the NEXT poll cycle
       only accumulates tuples tagged with the new checkpoint
  ELSE:
    sleep/backoff, poll again
```

**Diversity Sampler flag**: the original doc marks batch-size-64 as
`[NEEDS VALIDATION]` — too small a batch risks catastrophic forgetting if
it's dominated by one image type. Dev 3 should log the class-label
distribution per batch. Note: that label field lives on the frozen Tier 1
schema, not on `ExperienceTuple` — if per-batch diversity monitoring is
wanted, add a `label` field to the Tier 3-owned `WiggleCacheEntry` or
`ExperienceTuple` schema. That's a new field on a schema Tier 3 owns, so
it's safe, not a frozen-schema violation. Flag as v2, not blocking tonight.

---

## 5. Work Breakdown — 3 Developers

### Developer 1 — Telemetry Consumer, Sequence Integrity & Biometric Effort
**Owns**: Redis consumer entrypoint, `tier3.processed_annotations`,
`tier3.effort_population_stats`, sequence check, biometric effort engine,
Z-score normalization + sanity filter.
**Input**: `RedisEventEnvelope` off `telemetry:ingest`, `WiggleCacheEntry`
off `wiggle_cache:{seed}`.
**Output**: `NormalizedEffortScore` + accepted `RedisEventEnvelope`, handed
in-process to Dev 2's stage (same worker pipeline, function call — not a
second queue, to avoid unnecessary latency).

### Developer 2 — Geometric Reward, E-DRDE Evaluator & Replay Buffer
**Owns**: `tier3.replay_buffer`, `tier3.ppo_training_runs` DDL, Geometric
Delta Engine, E-DRDE Scalar Evaluator, State-Action-Reward Aggregator,
replay buffer writes.
**Input**: `NormalizedEffortScore` from Dev 1 (in-process), `M_final` from
the webhook `result`, `M_initial`/`M_wiggled` from `WiggleCacheEntry`.
**Output**: rows in `tier3.replay_buffer`.

### Developer 3 — Rollout Queue, PPO Training Loop & Blue/Green Deploy
**Owns**: Tier 4 entirely — poller, PPO Actor-Critic training step, LoRA
backprop, checkpoint packaging, Blue/Green swap via Traefik/Nginx,
`tier3.ppo_training_runs` writes (status/losses/model_version_out).
**Input**: `tier3.replay_buffer` (reads only rows Dev 2 writes; Dev 3 is the
only one allowed to flip `consumed_by_ppo`).
**Output**: updated `.safetensors` LoRA weights, hot-swapped into the
inference container Tier 2's `serving_ui` reads from.

**Local testing without teammates**: seed `tier3.replay_buffer` directly
with 64+ synthetic rows (script it) to test the PPO trigger without waiting
on Dev 1/Dev 2's live pipeline. Conversely, Dev 1/2 can fake a
`WiggleCacheEntry` + `RedisEventEnvelope` pair and push directly onto
`telemetry:ingest` with `redis-cli` to test without Tier 2 running.

---

## 6. Acceptance Criteria

- [ ] Consumer never crashes on a missing `WiggleCacheEntry` — logs and drops, keeps consuming
- [ ] Duplicate/out-of-order webhooks are provably rejected (test: replay the same `annotation_id` twice, confirm only one `replay_buffer` row)
- [ ] `tier3.replay_buffer` never mixes `model_version`s within a single PPO batch
- [ ] `tier3` schema tables only — zero queries against `tier1_predictions`, `junior_queue`, `senior_queue`, `consensus_queue`
- [ ] All DB/Redis calls are async (`asyncpg`/`redis.asyncio`); no `psycopg2` on the event loop
- [ ] A full 64-tuple batch triggers exactly one PPO run, writes one `ppo_training_runs` row, and results in a Blue/Green swap (can be a no-op/mock swap tonight, but the trigger + audit row must be real)
