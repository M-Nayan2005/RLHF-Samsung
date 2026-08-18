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

## 3. Ready-to-Use AI Coding Prompts

Each prompt assumes the assistant (Claude Code / Antigravity) has the repo
checked out and can read `common/schemas/*.py` and `.env.example` directly —
tell it to open those files first so it doesn't invent field names.

### Prompt — Developer 1 (Pre-Inference)
```
You're implementing services/pre_inference for a 4-tier RLHF segmentation
pipeline. Read common/schemas/tier1_ingestion.py first — that Pydantic module
is the frozen output contract, do not change field names or types.

Build a FastAPI service that:
1. Exposes POST /predict accepting {image_url: str, text_prompt: str}
2. Runs Grounding DINO to get a bounding box + label for the prompt
3. Runs SAM2 on that box 5 times with MC Dropout active (keep dropout layers
   in train mode during inference) to produce 5 MCDSample masks + class logits
4. Computes geometric_variance (spatial spread across the 5 polygon masks —
   use mean pairwise IoU distance or centroid/area variance, pick one and
   document it) and class_logit_entropy (Shannon entropy of the
   softmax-averaged class logits across the 5 samples)
5. Computes a consensus_mask (mean polygon, e.g. average vertex positions
   after alignment, or the sample closest to centroid — document your choice)
6. Returns a GroundedSAM2Output instance (use the Pydantic model directly,
   don't hand-roll the dict)
7. Persists the same record to a Postgres table `tier1_predictions` using
   DATABASE_URL from env

Use PyTorch, load checkpoints from GROUNDING_DINO_CHECKPOINT and
SAM2_CHECKPOINT env vars, respect DEVICE env var (cpu/cuda) so it runs on a
laptop without a GPU for local dev. Write a tests/mocks/tier1_output.json
fixture matching your actual output shape when you're done, and a README in
services/pre_inference/ documenting the variance/entropy formulas you chose.
```

### Prompt — Developer 2 (Routing, QA & Honeypot)
```
You're implementing services/routing_qa. Read common/schemas/routing_queue.py
(your output contract) and common/schemas/tier1_ingestion.py (your input
contract) first — don't change field names or types.

Build a FastAPI service that:
1. Polls the `tier1_predictions` Postgres table (created by pre_inference)
   for new/unrouted rows every few seconds
2. For each new row, applies dual-metric routing thresholds (read from env
   or a config table, don't hardcode): low variance + low entropy -> junior;
   high entropy (even with low variance — "confidently wrong") OR high
   variance -> senior; extreme combined score -> consensus
3. Applies the Stochastic Audit Filter: even "perfect" (would-be junior)
   tasks get routed to consensus_queue instead with probability
   STOCHASTIC_AUDIT_RATE (env, default 0.05)
4. Periodically injects a honeypot: pick a known verified mask, wrap it as a
   QueueTask with honeypot.is_honeypot=True and honeypot.ground_truth_mask
   set, push to junior_queue at rate HONEYPOT_INJECTION_RATE
5. Writes QueueTask rows to junior_queue/senior_queue/consensus_queue
   Postgres tables, and mirrors just the task_id onto the matching Redis
   list key (env: REDIS_JUNIOR_QUEUE_KEY etc.) via LPUSH
6. Exposes GET /tasks/next?queue=junior&annotator_id=X — atomically claims
   the oldest pending task in that queue (status pending->assigned,
   assigned_to=annotator_id), returns it as QueueTask. CRITICAL: never
   include honeypot.ground_truth_mask in this response even when
   is_honeypot=True — strip it before serializing.
7. Exposes POST /tasks/{task_id}/requeue — increments retry_count, if it
   exceeds MAX_CONSENSUS_RETRIES sets status=discarded and writes to a
   discard_bin table instead of re-queuing
8. Exposes POST /tasks/{task_id}/honeypot-result — internal endpoint Dev 4's
   pipeline (or a stub for now) calls when a honeypot task is submitted, to
   update the annotator's trust score in a `annotator_trust_scores` table

Write tests/mocks/routing_task.json as a canned QueueTask fixture other
devs can use to stub your API. Document your routing thresholds in a README.
```

### Prompt — Developer 3 (Interactive Serving & Stochastic Policy)
```
You're implementing services/serving_ui — a Label Studio ML Backend plus the
Label Studio project setup. Read common/schemas/routing_queue.py (your
input, from routing_qa) and common/schemas/label_studio_webhook.py (your
output, especially LSTelemetryMeta) first.

Build:
1. A Label Studio ML Backend (use the label-studio-ml-backend SDK) with a
   /predict method that: calls GET /tasks/next?queue=junior&annotator_id=...
   on routing_qa (or, if MOCK_MODE=true env var is set, reads
   tests/mocks/routing_task.json instead), takes baseline_mask, samples
   Gaussian noise N(0, WIGGLE_SIGMA) and applies it to the polygon vertices
   (document exactly how — perturb each vertex independently vs a shared
   affine jitter, pick one), returns an LS-compatible `predictions` block
   pre-populating the task with the wiggled polygon
2. A Label Studio labeling config (XML) for polygon segmentation on images
3. Custom LSF frontend instrumentation (JS) that listens for: polygon vertex
   add (click), vertex drag (accumulate Euclidean path length in px), and
   pointer dwell time over the boundary region. Before the annotation
   submits, write these into annotation.meta.effort_telemetry matching
   LSTelemetryMeta exactly (click_count, cursor_path_length_px,
   dwell_time_ms, wiggle_seed). If Label Studio's meta injection API doesn't
   support this cleanly, instead fire a separate POST to
   WEBHOOK_GATEWAY_URL/telemetry/raw with the same shape right before
   submit, and tell me which approach you used and why.
4. Configure the Label Studio project's webhook (ANNOTATION_UPDATED) to
   point at WEBHOOK_GATEWAY_URL/webhooks/label-studio

Document in services/serving_ui/README.md exactly which of the two
telemetry-capture approaches you implemented, since Dev 4 needs to know.
```

### Prompt — Developer 4 (Webhook Ingestion & Message Broker)
```
You're implementing services/webhook_gateway. Read
common/schemas/label_studio_webhook.py (your input contract) and
common/schemas/redis_event.py (your output contract) first.

Build a FastAPI service that:
1. Exposes POST /webhooks/label-studio: verify the request signature/secret
   against LABEL_STUDIO_WEBHOOK_SECRET (HMAC or shared-secret header,
   whichever Label Studio's webhook config supports), parse the body as
   LSAnnotationUpdatedPayload. If parsing fails, return 422 and log the raw
   body to a file/table for debugging (don't swallow it silently) —
   Dev 3 will need these logs if their telemetry shape is off.
2. Also expose POST /telemetry/raw accepting the same effort_telemetry
   shape directly, as a fallback transport in case Dev 3 can't get
   annotation.meta injection working (see their README once they ship it).
3. On successful parse, wrap in RedisEventEnvelope with a fresh UUID4
   event_id, idempotency_key = payload.annotation_id, enqueued_at = now
   (UTC ISO-8601), LPUSH the serialized envelope onto Redis list key
   REDIS_TELEMETRY_QUEUE_KEY. Must respond 200 within ~50ms — no
   synchronous Postgres writes in this request path.
4. Idempotency: before pushing, check a Redis SET (key e.g.
   "seen:annotation_ids", with a TTL) for idempotency_key; if present, skip
   the push and return 200 anyway (Label Studio may retry deliveries).
5. Write a tiny stub consumer script (separate process or a background
   task) that BRPOPs REDIS_TELEMETRY_QUEUE_KEY in a loop, deserializes as
   RedisEventEnvelope, and just logs "received {event_id} for annotation
   {idempotency_key}" — this proves the pipe works without building the
   real Tier 3 E-DRDE engine.

Write tests/mocks/ls_webhook_payload.json as a canned
LSAnnotationUpdatedPayload fixture, and document in a README how to curl it
against /webhooks/label-studio to verify the Redis push without Label
Studio or Dev 3's service running.
```

---

## 4. Git Collaboration & Integration Strategy

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
- [ ] Each `services/<name>/README.md` documents any deviation from the AI prompt above (esp. Dev 3's telemetry approach, Dev 1's variance formula)
