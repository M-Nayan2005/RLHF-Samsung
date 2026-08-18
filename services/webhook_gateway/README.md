# Tier 4 — Webhook Gateway & Message Broker

This service acts as the boundary between Label Studio and the downstream telemetry processing pipelines. It ingests webhooks, validates them against our frozen schema contracts, drops duplicates using Redis-based idempotency, and queues them up for Tier 3 consumption.

## Setup & Running Locally

Ensure you have Redis running:
```bash
docker compose up redis -d
```

### 1. Run the Gateway (Terminal 1)

If you are running outside of docker-compose, ensure `PYTHONPATH` includes the root directory so it can find `common.schemas`.

```bash
export PYTHONPATH=$(pwd)/../../
export REDIS_URL=redis://localhost:6379/0
export LABEL_STUDIO_WEBHOOK_SECRET=default_secret
export REDIS_TELEMETRY_QUEUE_KEY=telemetry:ingest

uvicorn services.webhook_gateway.main:app --reload --port 8004
```

### 2. Run the Stub Consumer (Terminal 2)

Because `BRPOP` is a blocking command, the consumer must be run as a separate process in its own terminal. It cannot share the FastAPI event loop.

```bash
export PYTHONPATH=$(pwd)/../../
export REDIS_URL=redis://localhost:6379/0
export REDIS_TELEMETRY_QUEUE_KEY=telemetry:ingest

python services/webhook_gateway/consumer.py
```

### 3. Test with Mock Payload (Terminal 3)

We have provided a mock payload that exactly matches the frozen `LSAnnotationUpdatedPayload` contract.

Send it to the webhook endpoint to simulate Label Studio:
```bash
curl -X POST http://localhost:8004/webhooks/label-studio \
  -H "Content-Type: application/json" \
  -H "Authorization: default_secret" \
  -d @tests/mocks/ls_webhook_payload.json
```

**Expected output:**
- Terminal 3 (curl) should return `{"status":"ok"}` in <50ms.
- Terminal 1 (Uvicorn) should log `Queued event <uuid> for annotation <annotation_id>`.
- Terminal 2 (Consumer) should log `received <uuid> for annotation <annotation_id>`.

If you run the curl command a second time within 24 hours, Terminal 1 will log `Duplicate webhook dropped for annotation <annotation_id>` and the consumer will not receive a second copy.
