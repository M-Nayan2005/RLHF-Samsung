import os
import uuid
import json
import logging
from datetime import datetime, timezone
from fastapi import FastAPI, Header, HTTPException, Request, Depends, status
from pydantic import ValidationError
import redis

from common.schemas.label_studio_webhook import LSAnnotationUpdatedPayload, LSTelemetryMeta
from common.schemas.redis_event import RedisEventEnvelope

app = FastAPI(title="Webhook Gateway")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_TELEMETRY_QUEUE_KEY = os.getenv("REDIS_TELEMETRY_QUEUE_KEY", "telemetry:ingest")
LABEL_STUDIO_WEBHOOK_SECRET = os.getenv("LABEL_STUDIO_WEBHOOK_SECRET", "default_secret")
IDEMPOTENCY_TTL_SECONDS = 86400

# Redis client setup
redis_client = None

@app.on_event("startup")
def startup_event():
    global redis_client
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    logger.info(f"Connected to Redis at {REDIS_URL}")

@app.on_event("shutdown")
def shutdown_event():
    if redis_client:
        redis_client.close()

def verify_secret(x_label_studio_signature: str = Header(None, alias="Authorization")):
    # Label Studio can send secrets in Authorization header or X-Label-Studio-Signature depending on config
    # We will check if the provided secret matches our environment variable
    if not x_label_studio_signature or x_label_studio_signature != LABEL_STUDIO_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid or missing webhook secret"
        )

@app.post("/webhooks/label-studio")
async def label_studio_webhook(request: Request, _ = Depends(verify_secret)):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    try:
        payload = LSAnnotationUpdatedPayload(**body)
    except ValidationError as e:
        logger.error(f"Schema mismatch. Raw payload: {json.dumps(body)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors())

    annotation_id = payload.annotation_id
    idempotency_key = f"seen:annotation_ids:{annotation_id}"

    # Idempotency check
    is_new = redis_client.set(idempotency_key, "1", ex=IDEMPOTENCY_TTL_SECONDS, nx=True)
    
    if not is_new:
        logger.info(f"Duplicate webhook dropped for annotation {annotation_id}")
        return {"status": "ok", "message": "duplicate dropped"}

    # Wrap into RedisEventEnvelope
    event_id = str(uuid.uuid4())
    enqueued_at = datetime.now(timezone.utc).isoformat()

    envelope = RedisEventEnvelope(
        event_id=event_id,
        idempotency_key=str(annotation_id),
        enqueued_at=enqueued_at,
        payload=payload
    )

    # Push to Redis
    redis_client.lpush(REDIS_TELEMETRY_QUEUE_KEY, envelope.model_dump_json())
    logger.info(f"Queued event {event_id} for annotation {annotation_id}")

    return {"status": "ok"}

@app.post("/telemetry/raw")
async def telemetry_raw(request: Request):
    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    task_id = body.get("task_id", "unknown_task")
    client_session_id = body.get("client_session_id", "unknown_session")
    
    # Generate fallback ID since annotation_id is null
    fallback_id = f"raw_{task_id}_{client_session_id}"
    idempotency_key = f"seen:annotation_ids:{fallback_id}"

    # Idempotency check
    is_new = redis_client.set(idempotency_key, "1", ex=IDEMPOTENCY_TTL_SECONDS, nx=True)
    if not is_new:
        logger.info(f"Duplicate raw telemetry dropped for fallback ID {fallback_id}")
        return {"status": "ok", "message": "duplicate dropped"}

    # Extract and validate effort_telemetry
    raw_telemetry = body.get("effort_telemetry", {})
    try:
        telemetry_meta = LSTelemetryMeta(**raw_telemetry)
    except ValidationError as e:
        logger.error(f"Invalid telemetry format: {json.dumps(raw_telemetry)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors())

    # Construct synthetic LSAnnotationUpdatedPayload
    from common.schemas.label_studio_webhook import LSAction
    synthetic_payload = LSAnnotationUpdatedPayload(
        action=LSAction.ANNOTATION_UPDATED,
        task_id=task_id,
        annotation_id=fallback_id,
        project_id=body.get("project_id", "unknown_project"),
        completed_by=body.get("completed_by", "unknown_annotator"),
        result=[],
        effort_telemetry=telemetry_meta,
        lead_time=body.get("lead_time", 0.0),
        created_at=datetime.now(timezone.utc).isoformat(),
        updated_at=datetime.now(timezone.utc).isoformat()
    )

    # Wrap into RedisEventEnvelope
    event_id = str(uuid.uuid4())
    enqueued_at = datetime.now(timezone.utc).isoformat()

    envelope = RedisEventEnvelope(
        event_id=event_id,
        idempotency_key=fallback_id,
        enqueued_at=enqueued_at,
        payload=synthetic_payload
    )

    # Push to Redis
    redis_client.lpush(REDIS_TELEMETRY_QUEUE_KEY, envelope.model_dump_json())
    logger.info(f"Queued event {event_id} for synthetic fallback annotation {fallback_id}")

    return {"status": "ok"}
