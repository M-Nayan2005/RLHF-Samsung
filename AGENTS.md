# AGENTS

This document defines the agent roles for the RLHF Segmentation Pipeline (Tier 1 & Tier 2). Future AI agents instantiated on this repository should self-select or be assigned one of these tracks.

## Roles & Responsibilities

### Track 1: Pre-Inference Agent
- **Domain**: ML inference (Grounding DINO, SAM2), geometric logic, database insertions.
- **Goal**: Read from inputs, run multiple forward passes, generate `geometric_variance` and `class_logit_entropy`, calculate a consensus mask, and persist `GroundedSAM2Output` to `tier1_predictions`.
- **Primary Files**: `services/pre_inference/*`, `common/schemas/tier1_ingestion.py`

### Track 2: Routing QA Agent
- **Domain**: Pipeline routing, threshold logic, queuing, stochastic sampling.
- **Goal**: Poll `tier1_predictions`, apply dual-metric routing rules (Junior vs Senior vs Consensus queues), inject honeypots occasionally, and expose HTTP endpoints for serving UI tasks. Manage task flow state in Postgres and Redis.
- **Primary Files**: `services/routing_qa/*`, `common/schemas/routing_queue.py`

### Track 3: Serving UI Agent
- **Domain**: Interactive annotation integration, ML backend serving, frontend telemetry.
- **Goal**: Serve tasks to Label Studio by injecting stochastic noise (wiggling masks), capture rich client-side frontend telemetry (mouse paths, click counts) via JS listeners, and integrate with the webhook payload.
- **Primary Files**: `services/serving_ui/*`, `common/schemas/label_studio_webhook.py`

### Track 4: Webhook Gateway Agent
- **Domain**: Ingestion, validation, message brokering.
- **Goal**: Fast, non-blocking webhook ingestion from Label Studio. Validate HMAC signatures, wrap data in a `RedisEventEnvelope`, enforce idempotency, and push to Redis queues.
- **Primary Files**: `services/webhook_gateway/*`, `common/schemas/redis_event.py`

## Rules of Engagement
- **Frozen Schemas**: NO agent may modify the files in `common/schemas/*` without cross-team (or user) approval.
- **Isolated Implementation**: Each agent must ONLY modify the files in its designated track directory (`services/<track>/`). Inter-service communication happens strictly through the shared schemas and defined HTTP/Redis paths.
- **Handoff Requirement**: Before halting or switching context, every agent must update `HANDOFF.md` to persist state for the next session.
