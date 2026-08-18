import os
import json
import logging
import redis

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_TELEMETRY_QUEUE_KEY = os.getenv("REDIS_TELEMETRY_QUEUE_KEY", "telemetry:ingest")

def run_consumer():
    logger.info(f"Connecting to Redis at {REDIS_URL}")
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    
    logger.info(f"Starting consumer, waiting for messages on '{REDIS_TELEMETRY_QUEUE_KEY}'...")
    
    while True:
        try:
            # BRPOP blocks until a message is available
            # It returns a tuple: (queue_name, message)
            result = redis_client.brpop(REDIS_TELEMETRY_QUEUE_KEY, timeout=0)
            if result:
                queue, message = result
                try:
                    envelope = json.loads(message)
                    event_id = envelope.get("event_id", "unknown_event")
                    idempotency_key = envelope.get("idempotency_key", "unknown_key")
                    logger.info(f"received {event_id} for annotation {idempotency_key}")
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode message: {message}")
        except redis.ConnectionError as e:
            logger.error(f"Redis connection error: {e}")
            # simple backoff could be added here
        except KeyboardInterrupt:
            logger.info("Consumer stopped.")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_consumer()
