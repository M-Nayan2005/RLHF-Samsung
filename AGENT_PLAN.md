# AGENT_PLAN

This is the comprehensive, exhaustive technical plan for the RLHF Tier 1 & Tier 2 pipeline. Agents must use this as their source of truth for implementation logic.

## Ground-Zero Assumptions
- **Auth**: MVP header stub (`X-Annotator-Id`), not real JWT auth. Swap later.
- **Telemetry**: Label Studio webhook lacks click/cursor data. Agent 3 must inject it client-side into `annotation.meta.effort_telemetry`. Fallback: separate `POST /telemetry/raw` endpoint.
- **Consensus queue (IAA)**: Multi-annotator UI is OUT OF SCOPE. Agent 2 just routes to it, logs it, and stops.

## Work Breakdown by Track

### Track 1: Pre-Inference & Auto-Labeling Engine
- Expose `POST /predict` taking `{image_url: str, text_prompt: str}`.
- Grounding DINO inference → bounding boxes + text labels.
- SAM2 forward pass, run **5x with MC Dropout active** (keep dropout layers in train mode during inference) → 5 `MCDSample`s.
- Compute `geometric_variance` (spatial spread across the 5 masks) and `class_logit_entropy` (entropy of averaged class logits).
- Compute `consensus_mask` (mean/representative polygon).
- Persist `GroundedSAM2Output` rows to Postgres table `tier1_predictions`.
- Return `GroundedSAM2Output`.

### Track 2: Routing, QA & Honeypot Engine
- Lightweight poller on `tier1_predictions` every N seconds.
- **Dual-metric router**: Low variance + low entropy -> junior; high entropy or high variance -> senior; extreme combined score -> consensus.
- **Stochastic Audit Filter**: `STOCHASTIC_AUDIT_RATE` (e.g. 5%) of perfect tasks forced into `consensus_queue`.
- **Honeypot injection**: Periodically inject known-gold task into `junior_queue` (`is_honeypot=True`). NEVER expose `ground_truth_mask` in API response.
- Expose `GET /tasks/next?queue=junior&annotator_id=...` → `QueueTask` (claims + marks assigned).
- Expose `POST /tasks/{task_id}/requeue` → limits retries via `MAX_CONSENSUS_RETRIES`, else moves to `discard_bin`.
- Expose `POST /tasks/{task_id}/honeypot-result` for trust score updates.

### Track 3: Interactive Serving & Stochastic Policy
- Label Studio ML Backend (`/predict`): Fetch `QueueTask` from Agent 2's `GET /tasks/next`, take `baseline_mask`, sample Gaussian noise `N(0, WIGGLE_SIGMA)`, return wiggled polygon as `predictions`.
- MOCK_MODE: If `MOCK_MODE=true`, read from `tests/mocks/routing_task.json` instead of polling Agent 2.
- **LSF Instrumentation**: Custom JS listeners on polygon tool (click, drag, hover). Write to `annotation.meta.effort_telemetry` before submit (fallback to `/telemetry/raw`).
- Setup Label Studio labeling config XML and webhook pointing to Agent 4's gateway.

### Track 4: Webhook Ingestion & Message Broker
- `POST /webhooks/label-studio`: Validate HMAC signature (`LABEL_STUDIO_WEBHOOK_SECRET`). Parse body as `LSAnnotationUpdatedPayload`. Return 422 if mismatched (log raw body).
- Also expose `POST /telemetry/raw` as fallback.
- Wrap into `RedisEventEnvelope` (`idempotency_key = annotation_id`). `LPUSH` onto `telemetry:ingest`.
- Must respond 200 OK in <50ms. No synchronous DB writes here.
- Idempotency check via Redis SET before pushing.
- **Stub Consumer**: Small worker `BRPOP`ing `telemetry:ingest` and logging receipt (proves pipeline works).
