"""
Tier 3 worker entrypoint: BRPOP `telemetry:ingest`, run Stage 1, hand off to Stage 2.

    python -m tier3_worker.consumer

Imports are package-relative: the container puts this code at
`/app/tier3_worker` (see the Dockerfile), so `services.tier3_worker` does not
exist there.
"""
import os
import sys
import asyncio
import logging
import asyncpg
import redis.asyncio as redis
from pydantic import ValidationError

from common.schemas.redis_event import RedisEventEnvelope
from .config import normalize_dsn
from .stage1 import process_envelope
from .stage2 import process_stage1, configure, get_writer, shutdown

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tier3_consumer")

QUEUE_KEY = os.getenv("REDIS_TELEMETRY_QUEUE_KEY", "telemetry:ingest")


async def worker(worker_id: int, redis_pool: redis.Redis, db_pool: asyncpg.Pool):
    logger.info(f"Worker {worker_id} started")
    while True:
        try:
            payload_bytes = None
            # 1. BRPOP telemetry:ingest
            result = await redis_pool.brpop(QUEUE_KEY, timeout=0)
            if not result:
                continue

            queue_name, payload_bytes = result

            # 2. Deserialize as RedisEventEnvelope
            try:
                envelope = RedisEventEnvelope.model_validate_json(payload_bytes)
            except ValidationError as e:
                logger.error(f"Worker {worker_id} dropped invalid payload: {e}")
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} failed to parse payload: {e}")
                continue

            # 3. Process Envelope (Dev 1 stage 1)
            stage1_output = await process_envelope(envelope, redis_pool, db_pool)

            # 4. Handoff to Dev 2 (Stage 2)
            if stage1_output:
                logger.info(f"Worker {worker_id} generated Stage1Output for {stage1_output.task_id}, handing off to Dev 2")
                await process_stage1(stage1_output)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Worker {worker_id} encountered unhandled exception: {e}")
            try:
                # Park the message back on the queue to avoid silent data loss
                if payload_bytes:
                    await redis_pool.lpush(QUEUE_KEY, payload_bytes)
            except Exception as push_err:
                logger.error(f"Worker {worker_id} failed to push back payload: {push_err}")
            await asyncio.sleep(1)


def database_dsn() -> str:
    """
    The same database Stage 2 writes to. `TIER3_PG_DSN` wins, as it does in
    `config.py`, so Stage 1's ledger and Stage 2's buffer can never end up in
    two different databases.
    """
    raw = os.getenv("TIER3_PG_DSN") or os.getenv("DATABASE_URL") or ""
    return normalize_dsn(raw) if raw else raw


async def main():
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    concurrency = int(os.getenv("TIER3_CONCURRENCY", "4"))

    redis_pool = redis.Redis.from_url(redis_url, decode_responses=True)

    # Stage 2's writer applies infra/sql/tier3_schema.sql on connect. It has to
    # happen before the first envelope: Stage 1 queries tier3.* tables first.
    configure()
    try:
        await get_writer().connect()
    except Exception as e:
        logger.error(f"Failed to connect to database / apply the tier3 schema: {e}")
        sys.exit(1)

    try:
        db_pool = await asyncpg.create_pool(dsn=database_dsn())
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        sys.exit(1)

    logger.info(f"Consuming {QUEUE_KEY} with {concurrency} workers")
    tasks = []
    for i in range(concurrency):
        tasks.append(asyncio.create_task(worker(i, redis_pool, db_pool)))

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        await shutdown()
        await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
