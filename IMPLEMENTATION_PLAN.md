# Tier 1 + Tier 2 — Tonight's Implementation Plan

Team: 4 developers, parallel tracks. Contracts are **frozen** — see `common/schemas/`.
Nobody edits another track's owned files without a PR + ping in #eng-contracts.

---

## 0. Ground-Zero Assumptions (confirm or override before 8pm)

- **Auth**: MVP header stub (`X-Annotator-Id`), not real JWT auth. Swap later.
- **Telemetry**: Label Studio's stock webhook has no click/cursor data. Dev 3 injects
  it client-side via LSF instrumentation into `annotation.meta.effort_telemetry`
  (see `common/schemas/label_studio_webhook.py::LSTelemetryMeta`). If Dev 3 can't
  get custom LSF meta injection working by tonight, the fallback is a **separate
  `POST /telemetry/raw` endpoint** fired by a small JS snippet in the Label Studio
  task template — same schema, different transport. Decide this by the first sync.
- **Tier 3/4 don't exist tonight**: `webhook_gateway` only needs to push a valid
  `RedisEventEnvelope` onto `telemetry:ingest` and log it. A stub consumer proves
  the queue works; the real E-DRDE math is out of scope.
- **Consensus queue (IAA)**: Dev 2 builds the routing/discard logic and the table,
  but the actual multi-annotator UI for Seniors is *not* built tonight — route to
  it, log it, stop there.

---

## 1. Repository Structure

```
repo/
├── .env.example
├── docker-compose.yml
├── common/
│   └── schemas/                # FROZEN — shared Pydantic contracts, all 4 devs import from here
│       ├── tier1_ingestion.py      # Dev 1 owns the values, everyone reads the shape
│       ├── routing_queue.py        # Dev 2 owns
│       ├── label_studio_webhook.py # Dev 3 owns
│       └── redis_event.py          # Dev 4 owns
├── services/
│   ├── pre_inference/           # Dev 1 — Grounding DINO + SAM2 5x MCD
│   ├── routing_qa/              # Dev 2 — Dual-metric router, honeypots, trust scores
│   ├── serving_ui/              # Dev 3 — Label Studio ML Backend, stochastic decoder
│   └── webhook_gateway/         # Dev 4 — FastAPI ingestion + Redis broker
└── tests/mocks/                 # Fixture payloads matching common/schemas, for local testing without teammates
```

**Rule**: a service may `import common.schemas.X`, but may NEVER import another
service's internal module directly. Cross-service communication only happens
through HTTP calls or the DB/Redis, using the shared schemas to (de)serialize.

---

## 2. Work Breakdown Structure

### Developer 1 — Pre-Inference & Auto-Labeling Engine
**Owns**: `services/pre_inference/*`, values in `common/schemas/tier1_ingestion.py`

- Grounding DINO inference → bounding boxes + text labels
- SAM2 forward pass, run **5x with MC Dropout active** → 5 `MCDSample`s
- Compute `geometric_variance` (spatial spread across the 5 masks) and
  `class_logit_entropy` (entropy of averaged class logits)
- Compute `consensus_mask` (mean/representative polygon) — this becomes `M_initial`
- Persist `GroundedSAM2Output` rows to Postgres table `tier1_predictions`
- Expose `POST /predict` (image_url, text_prompt) → `GroundedSAM2Output` for local testing

**Input interface**: raw image URL/path + text prompt (from ingestion CLI/script tonight — no upload UI needed)
**Output interface**: `GroundedSAM2Output` JSON, written to `tier1_predictions` table, also returnable via `POST /predict`

**Local testing without teammates**: `tests/mocks/tier1_output.json` — a canned
`GroundedSAM2Output` fixture. Run `pre_inference` standalone, hit `/predict`,
diff the response shape against the fixture and against `GroundedSAM2Output.schema()`.

---

### Developer 2 — Routing, QA & Honeypot Engine
**Owns**: `services/routing_qa/*`, `common/schemas/routing_queue.py`

- Dual-metric router: reads `geometric_variance` + `class_logit_entropy` from
  Tier 1 records, classifies into `junior_queue` / `senior_queue` / `consensus_queue`
  (thresholds as config, not hardcoded — read from env/config table)
- Stochastic Audit Filter: 5% of "perfect" (low-variance, low-entropy) tasks
  forced into `consensus_queue` anyway (`STOCHASTIC_AUDIT_RATE`)
- Honeypot injection: periodically inject a known-gold task into `junior_queue`
  with `HoneypotMeta.is_honeypot=True` — **never expose `ground_truth_mask` in
  any API response**, server-side comparison only
- Trust Score Engine: on a submitted honeypot task, compare submitted mask vs
  `ground_truth_mask`, bump/penalize `annotator_id`'s trust score in Postgres
- Max Retry Threshold: consensus tasks that exceed `MAX_CONSENSUS_RETRIES` get
  `status="discarded"`, written to a `discard_bin` table, not re-queued
- Expose `GET /tasks/next?queue=junior&annotator_id=...` → `QueueTask` (claims + marks `assigned`)
- Expose `POST /tasks/{task_id}/requeue` for the consensus retry path

**Input interface**: polls/subscribes to new `tier1_predictions` rows (simplest: a
lightweight poller every N seconds tonight, not a full event bus)
**Output interface**: `QueueTask` rows in `junior_queue`/`senior_queue`/`consensus_queue`
tables, mirrored into Redis lists (`REDIS_JUNIOR_QUEUE_KEY` etc, task_id only) so
Dev 3's serving layer can do fast existence/claim checks without hitting Postgres
on every poll.

**Local testing without teammates**: seed Postgres directly with 5–10 rows built
from `tests/mocks/tier1_output.json` (vary `geometric_variance`/`class_logit_entropy`
to hit all three routing branches), then hit `/tasks/next` and assert the right queue.

---

### Developer 3 — Interactive Serving & Stochastic Policy
**Owns**: `services/serving_ui/*`, Label Studio project config, `common/schemas/label_studio_webhook.py`

- Label Studio ML Backend (`/predict` endpoint LS calls when a task loads):
  fetch the `QueueTask` from Dev 2's `GET /tasks/next`, take `baseline_mask`,
  sample Gaussian noise `N(0, WIGGLE_SIGMA)` in latent/coordinate space, return
  a wiggled polygon as an LS-compatible `predictions` block (pre-annotate the task)
- LSF instrumentation: custom event listeners on the polygon tool (click,
  vertex-drag, hover-dwell) → accumulate `click_count`, `cursor_path_length_px`,
  `dwell_time_ms` → inject into `annotation.meta.effort_telemetry` before submit
  fires the LS webhook (see assumption #2 above for the fallback path)
- Label Studio project setup: labeling config XML, connect ML backend, connect
  webhook to Dev 4's gateway URL (`WEBHOOK_GATEWAY` env)

**Input interface**: `QueueTask` from Dev 2's `GET /tasks/next`
**Output interface**: Label Studio fires `ANNOTATION_UPDATED` webhook with
`result` + `effort_telemetry` block matching `LSAnnotationUpdatedPayload`, to
Dev 4's gateway URL

**Local testing without teammates**: `tests/mocks/routing_task.json` (a canned
`QueueTask`) stands in for Dev 2's API. Point the ML backend's task-fetch at a
local flag that serves the fixture instead of calling `routing_qa` when
`MOCK_MODE=true`. Verify wiggle output visually in Label Studio's preview.

---

### Developer 4 — Webhook Ingestion & Message Broker
**Owns**: `services/webhook_gateway/*`, `common/schemas/redis_event.py`

- `POST /webhooks/label-studio`: validate HMAC signature
  (`LABEL_STUDIO_WEBHOOK_SECRET`), parse body as `LSAnnotationUpdatedPayload`,
  reject with 422 on schema mismatch (log the raw payload for Dev 3 debugging)
- Wrap into `RedisEventEnvelope` (`idempotency_key = annotation_id`), `LPUSH`
  onto `telemetry:ingest`, return `200 OK` in <50ms (no synchronous DB writes here)
- Stub consumer: a small worker that `BRPOP`s `telemetry:ingest`, deserializes,
  logs `"received {event_id}"`, and that's it — proves the pipe end-to-end
  without building Tier 3
- Idempotency: if `idempotency_key` was already seen (keep a Redis SET of seen
  keys with TTL), drop duplicate webhook deliveries instead of double-queuing

**Input interface**: raw Label Studio webhook HTTP POST
**Output interface**: `RedisEventEnvelope` entries on `telemetry:ingest`

**Local testing without teammates**: `tests/mocks/ls_webhook_payload.json` — POST
it directly with `curl`/httpie against `/webhooks/label-studio` without needing
Label Studio or Dev 3 running. Verify it lands on the Redis list with `redis-cli
LRANGE telemetry:ingest 0 -1`.

---

## 3. Git Collaboration & Integration Strategy

### Branching
- `main` — always deployable, protected, PR-only
- `develop` — tonight's integration branch, everyone merges here first
- `feature/track-1-pre-inference` (Dev 1)
- `feature/track-2-routing-qa` (Dev 2)
- `feature/track-3-serving-ui` (Dev 3)
- `feature/track-4-webhook-gateway` (Dev 4)
- `common/schemas/*` changes get their own tiny PR (`chore/schema-<name>`),
  reviewed by whoever's downstream consumer, merged to `develop` before the
  4 tracks branch off — freeze it fast, then don't touch it again tonight
  unless everyone agrees in #eng-contracts.

### Local Integration Workflow
1. `cp .env.example .env`, fill in checkpoints paths / secrets
2. `docker compose up postgres redis` first — everyone's services need these
3. Each dev runs **only their own service** locally against real
   postgres/redis (`docker compose up pre_inference`, etc.) while stubbing
   upstream dependencies with `tests/mocks/*.json` per their track's "local
   testing" section above
4. Once two adjacent tracks are both ready (e.g. Dev 1 + Dev 2), do a pairwise
   integration test: bring both services up for real, turn off mocks, verify
   data flows through Postgres/Redis instead of fixtures
5. Final integration: `docker compose up --build` brings up all 6 containers;
   run the smoke test below

### Smoke Test (run before any merge to `develop`)
1. POST an image to `pre_inference` `/predict` → row appears in `tier1_predictions`
2. Within a few seconds, a `QueueTask` appears in `junior_queue` (check via
   `routing_qa`'s `/tasks/next`)
3. Open Label Studio, load the task, confirm a wiggled polygon pre-populates
4. Manually correct it, submit
5. `redis-cli LRANGE telemetry:ingest 0 -1` shows a new envelope; the stub
   consumer's logs show `"received {event_id}"`

### Acceptance Criteria for Merging Tonight
- [ ] All 4 services build and start via `docker compose up --build` with no manual patching
- [ ] `common/schemas` unchanged from the frozen versions (or changes were reviewed + merged to `develop` before 6pm)
- [ ] Each service's `/health` (or equivalent) endpoint returns 200
- [ ] The 5-step smoke test above passes end-to-end at least once
- [ ] No service crashes on malformed input from an upstream mock (422s, not 500s)
- [ ] Honeypot `ground_truth_mask` is never present in any `/tasks/next` HTTP response (grep the response body in the smoke test)
- [ ] Webhook gateway responds within ~50ms and doesn't block on Postgres
- [ ] Each `services/<name>/README.md` documents any deviation from the design specification above (esp. Dev 3's telemetry approach, Dev 1's variance formula)
