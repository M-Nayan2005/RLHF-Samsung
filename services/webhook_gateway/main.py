import os
import uuid
import json
import logging
import re
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

def verify_secret(x_label_studio_secret: str = Header(None)):
    if not x_label_studio_secret or x_label_studio_secret != LABEL_STUDIO_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Invalid or missing webhook secret"
        )

@app.post("/webhooks/label-studio")
async def label_studio_webhook(request: Request, _ = Depends(verify_secret)):
    try:
        body_bytes = await request.body()
        body_str = body_bytes.decode('utf-8')
        body = json.loads(body_str)
    except Exception:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    # Extract wiggle_seed via regex
    wiggle_seed = None
    match = re.search(r'seed=([a-fA-F0-9]+)', body_str)
    if match:
        wiggle_seed = match.group(1)
    elif "wiggle_seed=" in body_str:
        match2 = re.search(r'wiggle_seed=([a-fA-F0-9]+)', body_str)
        if match2:
            wiggle_seed = match2.group(1)
            
    task_id = body.get("task_id")

    # Fetch buffered beacon from Redis
    effort_telemetry = None
    if wiggle_seed:
        cached = redis_client.get(f"beacon:seed:{wiggle_seed}")
        if cached:
            effort_telemetry = json.loads(cached)
    
    if not effort_telemetry and task_id:
        cached = redis_client.get(f"beacon:task:{task_id}")
        if cached:
            effort_telemetry = json.loads(cached)
            
    if not effort_telemetry:
        logger.warning(f"No buffered beacon found for seed={wiggle_seed}, task={task_id}. Enqueuing with empty telemetry.")
        effort_telemetry = {
            "click_count": 0,
            "cursor_path_length_px": 0.0,
            "dwell_time_ms": 0,
            "wiggle_seed": wiggle_seed
        }

    # Inject effort_telemetry into the body to satisfy the frozen schema
    body["effort_telemetry"] = effort_telemetry

    try:
        payload = LSAnnotationUpdatedPayload(**body)
    except ValidationError as e:
        logger.error(f"Schema mismatch. Raw payload: {body_str}")
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
    logger.info(f"Queued event {event_id} for annotation {annotation_id} (joined with beacon seed={wiggle_seed})")

    return {"status": "ok"}

@app.post("/telemetry/raw")
async def telemetry_raw(request: Request):
    try:
        body_bytes = await request.body()
        body = json.loads(body_bytes)
    except json.JSONDecodeError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON")

    wiggle_seed = body.get("wiggle_seed")
    task_id = body.get("task_id")
    raw_telemetry = body.get("effort_telemetry")

    if not raw_telemetry:
        return {"status": "ok", "message": "no telemetry to buffer"}

    # Validate against LSTelemetryMeta
    try:
        LSTelemetryMeta(**raw_telemetry)
    except ValidationError as e:
        logger.error(f"Invalid telemetry format: {json.dumps(raw_telemetry)}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=e.errors())

    telemetry_json = json.dumps(raw_telemetry)
    
    # Buffer in Redis for 5 minutes (300 seconds)
    if wiggle_seed:
        redis_client.set(f"beacon:seed:{wiggle_seed}", telemetry_json, ex=300)
    if task_id:
        redis_client.set(f"beacon:task:{task_id}", telemetry_json, ex=300)

    logger.info(f"Buffered beacon for seed={wiggle_seed}, task={task_id}")
    return {"status": "ok"}
