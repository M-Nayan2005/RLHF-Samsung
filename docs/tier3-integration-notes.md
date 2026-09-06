# Tier 3 integration notes — Dev 2's seams

What Dev 2's half of `tier3_worker` needs from other people, and what it
produces for them. Three audiences: whoever writes the Tier 2 Redis cache,
Dev 1, and Dev 3.

Companion to [`integration-notes.md`](integration-notes.md) (Track 3, Tier 2)
and [`tier3-dev2-decisions.md`](tier3-dev2-decisions.md) (the reasoning).

---

## 1. For whoever owns `serving_ui` — the `wiggle_cache` writer

The Tier 3/4 plan §0 calls this the blocking dependency, and it is: without it
the geometric half of the reward cannot be computed. **It does not exist yet** —
there is no `wiggle_cache` anywhere in the repo, and `serving_ui` currently
makes no Redis calls at all.

Write one Redis key per served task, at the moment the wiggled mask goes out:

```
key    wiggle_cache:{wiggle_seed}
value  WiggleCacheEntry, JSON
TTL    ~24h
```

### The part that is easy to miss

`WiggleCacheEntry` now requires **`image_width`, `image_height` and
`model_version`** on top of the fields the plan listed. They are not optional
conveniences:

- **Dimensions.** Label Studio returns `M_final` as percentages of the image;
  the cached masks are absolute pixels. The conversion is anisotropic, so it
  cannot be guessed. Label Studio *does* send `original_width`/`original_height`
  on every region, and Tier 2 sets them — but the gateway re-parses the body
  through `LSResultRegion` (`id`/`type`/`value` only, pydantic ignores extras)
  and they are stripped before the envelope reaches Redis. This cache entry is
  the only surviving channel. Without them ΔIoU is not computable at all.
- **`model_version`.** `tier3.replay_buffer.model_version` is `NOT NULL`, and
  Tier 4 refuses to mix checkpoints inside one PPO batch. A tuple without one is
  unstorable and untrainable, and Dev 2 drops it.

### Good news: nothing new has to be computed

Every required value already sits on `ServedWiggleRecord`
(`services/serving_ui/app/models.py`), which `store.py` already persists per
served task. The writer is a projection:

```python
entry = WiggleCacheEntry(
    wiggle_seed=record.wiggle_seed,
    task_id=record.task_id,
    image_id=record.image_id,
    m_initial=PolygonMask(points=record.baseline_points),
    m_wiggled=PolygonMask(points=record.wiggled_points),
    served_at=record.served_at,
    image_width=record.image_width,       # already there
    image_height=record.image_height,     # already there
    model_version=record.model_version,   # already there
    is_honeypot=record.is_honeypot,       # already there, optional
    label=label,                          # optional, for Tier 4 diversity logging
)
await redis.set(f"wiggle_cache:{record.wiggle_seed}", entry.model_dump_json(), ex=86400)
```

Note `m_initial` takes `baseline_points` — the canonicalised μ — and `m_wiggled`
takes `wiggled_points`, the sampled action `A_t`. Getting these the wrong way
round inverts the reward, and nothing downstream can detect it.

Adding a Redis write does not violate either boundary rule: it is a new key, not
a Postgres table, and it modifies no frozen schema.

---

## 2. For Dev 1 — constructing `Stage1Output`

The handoff contract is unchanged in shape. Two required fields were added,
both copied straight off the cache entry you already fetch:

```python
return Stage1Output(
    annotation_id=envelope.payload.annotation_id,
    task_id=envelope.payload.task_id,
    wiggle_seed=entry.wiggle_seed,
    m_initial=entry.m_initial,
    m_wiggled=entry.m_wiggled,
    ls_result=[r.model_dump() for r in envelope.payload.result],
    effort=normalized,
    dropped_as_bot=normalized.dropped_as_bot,
    model_version=entry.model_version,

    image_width=entry.image_width,     # new, required
    image_height=entry.image_height,   # new, required
    is_honeypot=entry.is_honeypot,     # new, optional
    m_gold=entry.m_gold,               # new, optional
    label=entry.label,                 # new, optional
)
```

They are required rather than optional deliberately. A `Stage1Output` without
dimensions is not partially useful — Dev 2 can compute nothing from it. A
`ValidationError` naming `image_width` is a two-second fix; a silently inert
worker that drains the queue and writes no rewards is not.

### Calling into Dev 2

```python
from tier3_worker import stage2

stage2.configure()                      # once, at worker startup
await stage2.process_stage1(stage1)     # once per accepted annotation
await stage2.shutdown()                 # on the way out, closes the pool
```

`configure()` with no arguments reads the environment and builds the
Postgres-backed writer. Pass `writer=InMemoryReplayBuffer()` for a mock-mode run
with no database.

### What `process_stage1` does to your loop

It returns `None` on every drop — bot-flagged effort, missing `model_version`,
degenerate polygon, an annotation with no polygon regions, a duplicate — so one
unusable annotation never stops the queue.

It **raises** on a Postgres failure. That is not a property of the tuple in hand
and will apply equally to the next ten thousand, so the retry/backoff/park
decision is yours, not the reward stage's. `TIER3_RAISE_ON_WRITE_FAILURE=false`
inverts it if you would rather keep consuming and lose the rows.

### Where your settings go

`tier3_worker/config.py` has a marked section for the consumer keys — Redis URL,
queue name, worker fan-out, bot-velocity ceiling. The sections are kept apart so
the two additions merge without touching each other's lines.

---

## 3. For Dev 3 — what lands in `tier3.replay_buffer`

`infra/sql/tier3_schema.sql` holds all four `tier3` tables. The worker applies
it on startup (idempotent, `IF NOT EXISTS` throughout), following
`routing_qa/db.py`'s convention — the compose `postgres` service mounts no init
directory, so a `docker-entrypoint-initdb.d` script would only ever run against
a brand-new volume.

One row per accepted annotation. Beyond the columns the plan specified, two are
present that it did not:

| Column | Why |
| --- | --- |
| `label` | Plan §4's per-batch class-diversity logging. Nullable — `NULL` means the cache entry carried none. |
| `is_honeypot` | Honeypot rows use the strict two-term ΔIoU against a real gold mask; everything else uses the production fallback. The two are differently scaled, so you can tell them apart before mixing them in a batch. Open question Q20. |

`alpha` and `beta` are stored **per row**, not read from config at training
time. They are provisional and will be retuned, and a buffer spanning a retune
would otherwise silently mix two reward scales.

`state_s_t` and `action_a_t` are absolute-pixel polygons, the space the policy
emitted them in — not the normalised frame the IoU is computed in.

Your pull query is already indexed for:

```sql
SELECT ... FROM tier3.replay_buffer
WHERE consumed_by_ppo = FALSE AND model_version = $1
ORDER BY created_at ASC LIMIT 64
```

`idx_replay_buffer_pull` is a partial index on
`(model_version, consumed_by_ppo, created_at) WHERE consumed_by_ppo = FALSE`.

Dev 3 is the only writer of `consumed_by_ppo`, `consumed_at` and `ppo_batch_id`.

### Seeding without waiting on anyone

The plan's advice holds — insert synthetic rows directly. Every column Dev 3
reads is plain scalar or JSONB; no Dev 1 or Dev 2 code needs to run.

---

## 4. Verifying without the other services

Dev 2's suite is fully offline:

```bash
cd services/tier3_worker && python -m pytest      # 67 tests, ~1s
```

For the parts a unit test cannot reach — that the DDL applies, that
`ON CONFLICT` stops a duplicate at the database rather than only in the
in-memory double, that asyncpg accepts the DSN:

```bash
docker compose up -d postgres
docker compose --profile tier3 build tier3_worker
docker compose --profile tier3 run --rm tier3_worker python -m tier3_worker.selfcheck
```

It applies the schema, round-trips one tuple, replays it to prove the
constraint holds, checks `delta_iou` against a hand-computed value, and deletes
its probe rows. Exit 0 means everything passed.

That last check is not decoration: it is how the asyncpg `timestamptz` bug
(DEV2-11) was found. The in-memory double accepted the ISO string happily and
the real driver refused it.

---

## 5. Still open, and worth settling early

| # | Question | Who decides |
| --- | --- | --- |
| Q18 | Are `alpha=1.0` / `beta=0.3` acceptable, and who calibrates? | Whoever owns the reward design |
| Q19 | The plan's ΔIoU is the complement of the canonical one — which is intended? | Same |
| Q11 | Is the ΔIoU reference the consensus mask or the wiggled one? Default is wiggled; `TIER3_REFERENCE_MASK` flips it. | Same |
| Q20 | May honeypot and fallback tuples share a PPO batch? | Dev 3 |
| Q22 | Widen `LSResultRegion` instead of extending the cache entry? Would fix the dimensions and region-identity matching at source. | Dev 3 (Track 3) + Dev 4 |

Full reasoning in [`tier3-dev2-decisions.md`](tier3-dev2-decisions.md).
