# tier3_worker — Tier 3 reward pipeline

Turns one human annotation into one row of PPO training data.

The service is split between two developers and this README covers **Dev 2's
half**: everything from a `Stage1Output` to a `tier3.replay_buffer` row.

```
Redis telemetry:ingest
   │
   ├─ [Dev 1]  sequence check → wiggle cache fetch → biometric effort → Z-score
   │           produces Stage1Output
   ▼
   ├─ [Dev 2]  Geometric Delta Engine       reward.compute_geometric_delta
   │           E-DRDE Scalar Evaluator      reward.evaluate_edrde
   │           State-Action-Reward Aggregator  aggregator.build_experience_tuple
   │           Offline Replay Buffer write  buffer.PostgresReplayBuffer
   ▼
tier3.replay_buffer  →  [Dev 3] Tier 4 poller, PPO, blue/green
```

## The entrypoint

```python
from tier3_worker import stage2

stage2.configure()                    # once, at worker startup
await stage2.process_stage1(stage1)   # once per accepted annotation
await stage2.shutdown()               # closes the pool
```

`process_stage1` holds exactly the signature the handoff contract specifies:
`async def process_stage1(stage1: Stage1Output) -> None`.

## Two things Dev 1 needs to know

**1. `Stage1Output` gained two required fields: `image_width`, `image_height`.**
Copy them straight off the `WiggleCacheEntry` you already fetch.

```python
Stage1Output(
    ...,
    image_width=entry.image_width,
    image_height=entry.image_height,
    model_version=entry.model_version,
)
```

They are required rather than optional because Dev 2 cannot compute ΔIoU
without them and cannot recover them from anywhere else — a `Stage1Output`
lacking them is not partially useful, it is inert. Why they are needed is in
[the contract file](../../common/schemas/tier3_rlhf.py) and summarised below.

**2. `WiggleCacheEntry` gained `image_width`, `image_height` and
`model_version`, all required.** Whoever writes the Tier 2 Redis cache must
populate them. All three already exist on `ServedWiggleRecord` in
`services/serving_ui/app/models.py`, so the cache writer is a projection of a
record Tier 2 already keeps — nothing new has to be computed.

## Why the image dimensions are load-bearing

Label Studio returns polygon vertices as **percentages of the image, 0–100**.
`m_initial` and `m_wiggled` are **absolute pixels**. An IoU between the two is
meaningless, and the conversion between them — `diag(100/W, 100/H)` — is
anisotropic, so it cannot be guessed or cancelled out.

Label Studio does put `original_width` / `original_height` on every region, and
Tier 2 sets them. **They do not reach Tier 3.** `services/webhook_gateway/main.py`
re-parses the body through `LSAnnotationUpdatedPayload`, whose `LSResultRegion`
declares only `id`, `type` and `value`; pydantic v2 ignores extras, so
`original_width`, `original_height` and `region.meta` are stripped before the
envelope is pushed onto `telemetry:ingest`. The cache entry is the only
surviving channel.

The masks are compared in a normalised unit frame (`x/W`, `y/H` for pixels,
`p/100` for percentages). IoU is invariant under that map — it is affine, so
intersection and union areas scale by the same determinant — which makes the
unit-frame IoU exactly equal to the pixel-frame one. That invariance is
asserted directly in `tests/test_geometry.py`.

## The reward

From [`docs/reference/equations.md`](../../docs/reference/equations.md), which
is the only trustworthy source — the `.docx` extracts flatten OMML math into
unreadable runs.

```
R_t   = alpha * ΔIoU - beta * ΔE_norm

ΔIoU  = IoU(M_final, M_gold) - IoU(M_ref, M_gold)     honeypot tasks
ΔIoU  = 1 - IoU(M_ref, M_final)                       production fallback
```

`M_ref` defaults to **`m_wiggled`**, the sampled action, not the consensus
mask. Open question Q11: reward has to attribute to the action taken, or every
action on a given task earns an identical score and the critic has nothing to
separate. `TIER3_REFERENCE_MASK=initial` flips it without a code change, for
when Q11 is formally settled.

`ΔE_norm` arrives already Z-scored from Dev 1 and is **not** re-normalised
here. Applying beta to a raw ΔE is the "scale disparity" flaw `tier3.docx`
names explicitly.

## Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `TIER3_ALPHA` | `1.0` | Weight on the accuracy term. **Provisional.** |
| `TIER3_BETA` | `0.3` | Weight on the effort penalty. **Provisional.** |
| `TIER3_REFERENCE_MASK` | `wiggled` | `wiggled` or `initial` — see Q11 above. |
| `TIER3_USE_GOLD_WHEN_AVAILABLE` | `true` | Use the strict ΔIoU on honeypot tasks. |
| `TIER3_PG_DSN` | falls back to `DATABASE_URL` | SQLAlchemy `+asyncpg` suffixes are stripped automatically. |
| `TIER3_APPLY_SCHEMA_ON_START` | `true` | Apply `infra/sql/tier3_schema.sql` on connect. Idempotent. |
| `TIER3_REQUIRE_MODEL_VERSION` | `true` | Drop tuples with no `model_version`. |
| `TIER3_RAISE_ON_WRITE_FAILURE` | `true` | Re-raise DB failures instead of losing rows silently. |

**alpha and beta are not spec-derived.** `docs/reference/equations.md` is
explicit that alpha, beta and w1..w3 "are never assigned anywhere" in the source
documents. These are the starting points the Tier 3/4 plan nominates. The worker
logs a WARNING naming them at every startup so no run is ever silently
calibrated. Filed as **Q18**.

## Dropped versus raised

Confusing these is how a pipeline either dies on one bad polygon or silently
discards a day of training data.

| Situation | Behaviour |
| --- | --- |
| Bot-flagged effort | Logged, dropped, loop continues |
| Missing `model_version` | Logged at ERROR, dropped |
| Degenerate / unrepairable polygon | Logged at ERROR, dropped |
| Annotation with no polygon regions | Logged at ERROR, dropped |
| Duplicate `annotation_id` | `ON CONFLICT DO NOTHING`, logged at INFO |
| **Postgres unreachable** | Logged and **re-raised** |

A database outage is not a property of the tuple in hand and will apply equally
to the next ten thousand, so it is the caller's decision, not this stage's.

## Tests

```bash
cd services/tier3_worker
python -m pytest            # 60 tests, fully offline, ~1s
```

No Postgres, no Redis, no network. The expected reward numbers are worked out
by hand in the test comments rather than captured from a run, so a regression
reads as a disagreement with the arithmetic.

For the parts a unit test cannot reach — that the DDL applies, that
`ON CONFLICT` stops a duplicate at the database rather than only in the
in-memory double, that asyncpg accepts the DSN:

```bash
docker compose --profile tier3 build tier3_worker
docker compose run --rm tier3_worker python -m tier3_worker.selfcheck
```

Idempotent and safe against a live database; it deletes its own probe rows.

## Ownership

| Path | Owner |
| --- | --- |
| `geometry.py`, `reward.py`, `aggregator.py`, `buffer.py`, `stage2.py`, `selfcheck.py` | Dev 2 |
| `config.py` | Dev 2's keys; Dev 1's Redis keys merge into the marked section |
| `consumer.py` (not yet present) | Dev 1 |
| `infra/sql/tier3_schema.sql` | Dev 2 owns `replay_buffer` + `ppo_training_runs`; Dev 1 owns the other two tables |
| `common/schemas/tier3_rlhf.py` | Shared. Additive changes only, documented in the header. |
