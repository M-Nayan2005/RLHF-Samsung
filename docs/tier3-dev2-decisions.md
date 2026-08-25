# Tier 3 — Developer 2 decisions

Every judgment call made building the Geometric Delta Engine, the E-DRDE Scalar
Evaluator, the State-Action-Reward Aggregator and the replay-buffer writer.
Companion to [`dev3-decisions.md`](dev3-decisions.md), which covers Track 3
(Tier 2 serving).

Scope: `services/tier3_worker/{geometry,reward,aggregator,buffer,stage2,config,selfcheck}.py`,
`infra/sql/tier3_schema.sql`, and the additive parts of
`common/schemas/tier3_rlhf.py`.

Numbering continues from Track 3: divergences resume at **D18**, open questions
at **Q18**. Decisions use a `DEV2-n` prefix so they cannot be confused with
Track 3's `DD-n`.

---

## Decisions

### DEV2-1 — IoU is exact (shapely/GEOS), not rasterised

`services/serving_ui/app/geometry.py` already has a `raster_iou`, and its own
docstring bars this use: *"This is NOT the Tier 3 delta-IoU: that one is exact,
lives in the E-DRDE engine... Do not import this from a reward path."* It
samples a 256×256 grid, which is fine for asserting a wiggle moved a mask and
not fine for a number that gets multiplied by `alpha` and backpropagated.

`shapely>=2` is therefore a dependency of this service. Rolling a polygon
clipper by hand was considered and rejected: exact boolean ops on concave,
self-intersecting rings are where subtle numerical bugs live, and a subtly
wrong reward is invisible until a model has trained on it.

### DEV2-2 — Every mask is compared in a normalised unit frame

Three masks meet in the reward and they do not arrive in the same units:
`m_initial` / `m_wiggled` / `m_gold` are absolute pixels; `M_final`, parsed from
the webhook, is percentages of the image (0–100).

All four are converted to a unit frame — `x/W, y/H` for pixels, `p/100` for
percentages — before any IoU is taken. That is legitimate because the
pixel→percent map is `diag(100/W, 100/H)`: affine and invertible, so
intersection and union areas scale by the same determinant and IoU is
unchanged. The unit-frame IoU is therefore *exactly* the pixel-frame IoU, not
an approximation of it. Asserted directly in
`tests/test_geometry.py::test_iou_is_invariant_under_the_percent_conversion`.

`state_s_t` and `action_a_t` are still **stored** in absolute pixels. Tier 4 has
to reconstruct the action in the space the policy emitted it, and a unit-frame
polygon stripped of its dimensions cannot be scaled back.

### DEV2-3 — ΔIoU is measured against the wiggled mask, and it is configurable

Open question Q11 / divergence D10. The frozen schemas label the Tier 1
consensus mask `M_initial`, but the mask the human actually corrected is the
wiggled one, and in RL terms that wiggled polygon is the sampled action `A_t`.
Reward must attribute to the action taken rather than to the mean of the
distribution it was sampled from — otherwise every action on a given task earns
an identical reward and the critic has nothing to separate.

Default: `m_wiggled`. `TIER3_REFERENCE_MASK=initial` flips it with no code
change, so when Q11 is formally settled the answer is a config edit. Track 3
de-risked this by persisting every served polygon; this consumes that work.

### DEV2-4 — The two-term `GeometricDelta` and the one-term canonical fallback are the same number

`GeometricDelta` (from the plan) requires `iou_initial` and `iou_final`, both in
[0, 1]. `docs/reference/equations.md` gives the production fallback as a single
term, `ΔIoU = 1 − IoU(M_initial, M_final)`. These look incompatible.

They are not. Taking `M_final` as the ground-truth proxy — a human looked at it
and signed off, so it is the best available estimate of truth for that image:

```
iou_initial = IoU(M_ref, M_final)
iou_final   = IoU(M_final, M_final) = 1.0
delta_iou   = iou_final − iou_initial = 1 − IoU(M_ref, M_final)
```

which is the canonical fallback exactly. On honeypot tasks `m_gold` replaces
the proxy and both terms carry independent information. One code path, two
references, **no schema edited and no number invented**.

### DEV2-5 — `WiggleCacheEntry` and `Stage1Output` gained required fields

Additive only; the plan explicitly permits new fields on Tier 3-owned schemas
(§4). See D18/D19 below for why they are not optional.

| Schema | Added | Required |
| --- | --- | --- |
| `WiggleCacheEntry` | `image_width`, `image_height`, `model_version` | yes |
| `WiggleCacheEntry` | `is_honeypot`, `m_gold`, `label` | no |
| `Stage1Output` | `image_width`, `image_height` | yes |
| `Stage1Output` | `is_honeypot`, `m_gold`, `label` | no |

Required rather than optional because a `Stage1Output` without dimensions is
not partially useful, it is inert — Dev 2 can compute nothing from it. A
pydantic `ValidationError` naming `image_width` at construction is a two-second
fix; the alternative is a worker that drains the queue and silently produces no
rewards at all.

All three required values already exist on Tier 2's `ServedWiggleRecord`
(`services/serving_ui/app/models.py`), so the Redis cache writer is a
projection of a record Tier 2 already keeps.

### DEV2-6 — `alpha` and `beta` are read from the environment and announced as provisional

`AGENTS.md` forbids inventing an unset hyperparameter, and
`docs/reference/equations.md` states that `alpha`, `beta` and `w1..w3` "are
never assigned anywhere" in the sources. The Tier 3/4 plan nonetheless nominates
`alpha=1.0, beta=0.3` as starting points.

Both are honoured: the plan's values are used, but they are read from
`TIER3_ALPHA` / `TIER3_BETA` rather than hardcoded into the arithmetic, and
`log_provisional_hyperparameters()` emits a **WARNING** naming them at every
startup. Each `replay_buffer` row also stores the pair it was computed with, so
a buffer spanning a retune stays interpretable instead of silently mixing two
reward scales. Filed as Q18.

### DEV2-7 — Data-shaped failures are dropped; infrastructure failures are raised

Conflating these is how a pipeline either dies on one bad polygon or silently
discards a day of training data.

Dropped, logged, loop continues: bot-flagged effort, missing `model_version`,
degenerate or unrepairable polygon, an annotation with no polygon regions,
duplicate `annotation_id`.

Re-raised: a Postgres failure. It is not a property of the tuple in hand and
will apply equally to the next ten thousand, so it is the caller's decision.
`TIER3_RAISE_ON_WRITE_FAILURE=false` inverts it if operational preference
differs.

### DEV2-8 — Idempotency is enforced at the database constraint, not only in code

Dev 1's `tier3.processed_annotations` ledger is the first line of defence. The
writer adds `ON CONFLICT (annotation_id) DO NOTHING` against the UNIQUE
constraint as the second, because the two run in separate transactions and two
workers racing a redelivery can both clear the ledger check. Only the
constraint can actually stop the duplicate row — which is what makes the
acceptance criterion ("replay the same `annotation_id`, confirm one row")
meaningful. Verified against real Postgres, not only against the in-memory
double.

### DEV2-9 — The name `replay_buffer` is kept; the behaviour is the ratified design

`docs/authority.md` **C1 rejects the "Offline Replay Buffer" by name**: PPO is
strictly on-policy and training on tuples from superseded weights breaks the
importance-sampling ratio. The plan and DDL nonetheless call the table
`replay_buffer`.

What the table actually implements is the ratified streaming-rollout design —
one `model_version` per batch, consume-once, no random resampling over history.
The behaviour is correct; only the name is inherited from the rejected design.
Kept so code, DDL and plan agree, and recorded here (and in the DDL, and on the
`ExperienceTuple` docstring) so nobody concludes C1 was reopened. Flagged as D21.

### DEV2-10 — The worker applies its own schema on startup

`services/routing_qa/db.py` already creates its tables in code, so this follows
the repo's own convention. The alternative — a `docker-entrypoint-initdb.d`
mount — runs only against a brand-new volume, and the compose `postgres`
service keeps a named `pgdata` volume, so it would silently skip every existing
deployment. Every statement in `tier3_schema.sql` is `IF NOT EXISTS` /
`ON CONFLICT`, so re-applying is a no-op. `TIER3_APPLY_SCHEMA_ON_START=false`
turns it off where migrations are managed elsewhere.

### DEV2-11 — `created_at` is coerced to a `datetime` before binding

`ExperienceTuple.created_at` is a string, as every timestamp in the frozen
contracts is. asyncpg refuses a string for a `timestamptz` parameter — it
encodes arguments client-side and raises `DataError: expected a datetime.date
or datetime.datetime instance` — and the `$13::timestamptz` cast cannot help
because the bind never reaches the server. `buffer.as_timestamptz` converts at
the boundary, reading a naive timestamp as UTC rather than local time so
`created_at` ordering (which Tier 4 batches on) is not skewed.

Found only by running `selfcheck` against a live Postgres; the in-memory double
accepted the string happily. Pinned offline in `tests/test_buffer.py`.

### DEV2-12 — `M_final` region selection prefers the served label, then unions

Regions are selected in order: those carrying the served `label`, else the sole
polygon region, else the union of all polygon regions. An annotator splitting
one object into two polygons meant their union; a second object with a
different label is not part of this region's correction cost.

Region *identity* would discriminate better, but it does not survive — see D18.

### DEV2-13 — The compose service sits behind a `tier3` profile

The image's default `CMD` is Dev 1's consumer entrypoint, which does not exist
yet. An unprofiled service would crash-loop for everyone running
`docker compose up`. Drop the `profiles` key once `tier3_worker/consumer.py`
lands.

### DEV2-14 — The DSN is normalised before asyncpg sees it

`.env.example` ships `DATABASE_URL=postgresql+asyncpg://...`, SQLAlchemy's
spelling. `asyncpg.create_pool` parses the scheme itself and rejects a
`+driver` suffix, so a worker inheriting the shared variable would fail to
connect for a reason unrelated to its credentials. `config.normalize_dsn`
strips it.

---

## Divergences — spec versus code

### D18 — The gateway strips `original_width`, `original_height` and `region.meta` · **MISMATCH**

Label Studio puts `original_width` / `original_height` on every region, and
Tier 2's `ls_format.build_prediction` sets them along with
`meta.text = "wiggle_seed=..."`. None of it reaches Tier 3.

`services/webhook_gateway/main.py:93` re-parses the request body through
`LSAnnotationUpdatedPayload`, whose `LSResultRegion` declares only `id`, `type`
and `value`. Pydantic v2 defaults to `extra="ignore"`, so the other keys are
dropped before `envelope.model_dump_json()` pushes onto `telemetry:ingest`.

Two consequences: the image dimensions must come from the Redis cache entry
(D19), and `M_final` cannot be matched to the served region by identity
(DEV2-12). Fixable by widening `LSResultRegion`, but that is a frozen Tier 2
contract owned by Dev 3 and consumed by Dev 4 — raised as Q22 rather than
edited.

### D19 — `WiggleCacheEntry` as specified cannot support an IoU · **MISMATCH**

The plan's `WiggleCacheEntry` carries the two polygons but no image dimensions.
`M_final` arrives in percentages and the cached masks are in pixels; the map
between them is anisotropic, so it cannot be guessed or cancelled. With D18
closing the webhook route, the cache entry is the only surviving channel, and
without dimensions ΔIoU is not computable **at all** — not merely
less accurate. Resolved by DEV2-5.

The same entry also lacked `model_version`, which the plan itself records as a
"KNOWN GAP" on `Stage1Output` while `tier3.replay_buffer.model_version` is
`NOT NULL` and Tier 4 batches on it. Also resolved by DEV2-5.

### D20 — The plan's ΔIoU is the complement of the canonical one · **MISMATCH**

The Tier 3/4 plan states `Delta_IoU = IoU(M_final, M_initial)`.
`docs/reference/equations.md`, reconstructed from `tier3.docx` and ratified by
`docs/authority.md` C2, gives the production fallback as
`ΔIoU = 1 − IoU(M_initial, M_final)` — the complement, i.e. the opposite sign
of improvement.

The canonical form is implemented, per the precedence rules. The difference is
not cosmetic: under the plan's version the reward rises when the human barely
changed the mask, under the canonical version it rises with the size of the
correction, and the two train toward opposite behaviours. Raised as Q19.

### D21 — `tier3.replay_buffer` contradicts authority C1 by name · **MISMATCH (naming only)**

See DEV2-9. Behaviour matches the ratified streaming-rollout design; only the
identifier is inherited from the rejected one.

### D22 — `alpha`, `beta` and `w1..w3` are assigned, though no source assigns them · **NEW**

The plan nominates `alpha=1.0, beta=0.3, w1=1.0, w2=0.01, w3=0.001` and
`EffortWeights` carries the `w` values as pydantic defaults. The sources assign
none of them. Mitigated rather than resolved by DEV2-6. Raised as Q18.

---

## Open questions

### Q18 — Are `alpha=1.0` and `beta=0.3` acceptable, and who calibrates them?

They set the entire trade-off between accuracy gain and human effort, and they
are nobody's measured values. The scale disparity `tier3.docx` warns about is
handled by Z-scoring `ΔE`, but the *ratio* of the two terms is still a free
choice. Needed before any run whose output is trusted. Related: D22.

### Q19 — Which ΔIoU sign convention is intended?

The canonical `1 − IoU(M_ref, M_final)` rises with the size of the correction,
so a policy is rewarded for masks the human had to change a lot — held in check
only by the `beta · ΔE_norm` penalty. The quantity being maximised is therefore
"geometric improvement per unit of human effort", which is coherent and matches
the stated purpose of E-DRDE. But it should be confirmed as intended rather
than inferred, because the plan states the opposite (D20) and the two train
toward opposite behaviours. Whichever way it is settled, the fix is one line.

### Q20 — May honeypot and fallback tuples share a PPO batch?

Extends Q6. The gold form and the proxy form are differently scaled quantities.
`replay_buffer.is_honeypot` now records which produced each row, so Tier 4
*can* separate them — but whether it *should* is unanswered.

### Q21 — How should `M_final` be chosen when the annotator draws several regions?

DEV2-12 picks label-match, then sole region, then union. Defensible, but it is a
policy choice with no source behind it, and it changes the reward when an
annotator splits an object or draws a neighbouring one. Region identity would
settle it if D18 were fixed.

### Q22 — Should `LSResultRegion` be widened instead of extending the cache entry?

Adding `original_width`, `original_height` and `meta` to `LSResultRegion` would
fix D18 at the source, give Tier 3 the dimensions for free, and restore
region-identity matching for Q21. It is a frozen Tier 2 contract owned by Dev 3
and consumed by Dev 4, so it was not touched. Worth deciding before the cache
writer is built, since the two solutions overlap.
