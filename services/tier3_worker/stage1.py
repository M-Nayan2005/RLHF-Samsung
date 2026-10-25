import os
import logging
from typing import Optional
from datetime import datetime, timezone
import asyncpg
import redis.asyncio as redis

from common.schemas.redis_event import RedisEventEnvelope
from common.schemas.tier3_rlhf import (
    Stage1Output, WiggleCacheEntry, NormalizedEffortScore
)

logger = logging.getLogger("tier3_stage1")

def derive_sequence_no(enqueued_at: str) -> int:
    try:
        dt = datetime.fromisoformat(enqueued_at.replace("Z", "+00:00"))
        return int(dt.timestamp() * 1000)
    except ValueError:
        return int(datetime.now(timezone.utc).timestamp() * 1000)

async def process_envelope(envelope: RedisEventEnvelope, redis_pool: redis.Redis, db_pool: asyncpg.Pool) -> Optional[Stage1Output]:
    payload = envelope.payload
    sequence_no = derive_sequence_no(envelope.enqueued_at)
    
    async with db_pool.acquire() as conn:
        prior_seq = await conn.fetchval(
            "SELECT MAX(sequence_no) FROM tier3.processed_annotations WHERE task_id = $1", 
            payload.task_id
        )
        if prior_seq is not None and sequence_no < prior_seq:
            logger.warning(f"Out of order sequence for task {payload.task_id} (seq {sequence_no} < prior {prior_seq})")
            return None
            
        insert_query = """
            INSERT INTO tier3.processed_annotations 
            (annotation_id, task_id, event_id, sequence_no, accepted, reject_reason)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (annotation_id) DO NOTHING
            RETURNING annotation_id
        """
        inserted = await conn.fetchval(
            insert_query, 
            payload.annotation_id, payload.task_id, envelope.event_id, sequence_no, True, None
        )
        if not inserted:
            logger.warning(f"Duplicate annotation_id {payload.annotation_id}")
            return None

    wiggle_seed = payload.effort_telemetry.wiggle_seed
    if not wiggle_seed:
        return None

    raw_cache = await redis_pool.get(f"wiggle_cache:{wiggle_seed}")
    if not raw_cache:
        logger.error(f"Missing WiggleCacheEntry for task_id={payload.task_id}")
        return None
        
    try:
        cache_entry = WiggleCacheEntry.model_validate_json(raw_cache)
    except Exception as e:
        logger.error(f"Failed to parse WiggleCacheEntry: {e}")
        return None

    click_count = payload.effort_telemetry.click_count
    cursor_path = payload.effort_telemetry.cursor_path_length_px
    dwell_time = payload.effort_telemetry.dwell_time_ms

    # The gateway still enqueues an annotation whose beacon never arrived, with
    # all-zero effort (docs/integration-notes.md). That is "no measurement",
    # not "no effort": rewarding it as zero effort would teach the policy that
    # whatever it served was free to correct. It also must not enter the
    # population stats below.
    if click_count == 0 and cursor_path == 0 and dwell_time == 0:
        logger.warning(f"DROP task={payload.task_id} annotation={payload.annotation_id} reason=no_telemetry")
        return None

    w1 = float(os.getenv("EDRDE_W1_CLICKS", "1.0"))
    w2 = float(os.getenv("EDRDE_W2_PATH", "0.01"))
    w3 = float(os.getenv("EDRDE_W3_DWELL", "0.001"))
    
    delta_e_raw = (w1 * click_count) + (w2 * cursor_path) + (w3 * dwell_time)

    async with db_pool.acquire() as conn:
        welford_query = """
            UPDATE tier3.effort_population_stats
            SET count = count + 1,
                mean = mean + ($1 - mean) / (count + 1),
                m2 = m2 + ($1 - mean) * ($1 - (mean + ($1 - mean) / (count + 1))),
                updated_at = now()
            WHERE id = 1
            RETURNING count, mean, m2
        """
        stats = await conn.fetchrow(welford_query, delta_e_raw)
        
        if stats and stats['count'] > 1:
            mean = stats['mean']
            variance = stats['m2'] / stats['count']
            stddev = variance ** 0.5
        else:
            mean = delta_e_raw
            stddev = 1.0

    epsilon = 1e-5
    safe_stddev = max(stddev, epsilon)
    delta_e_norm = (delta_e_raw - mean) / safe_stddev

    bot_threshold = float(os.getenv("EDRDE_BOT_VELOCITY_PX_MS", "5.0"))
    safe_dwell = max(dwell_time, 1)
    velocity = cursor_path / safe_dwell
    dropped_as_bot = (velocity > bot_threshold)

    effort = NormalizedEffortScore(
        delta_e_raw=delta_e_raw,
        delta_e_norm=delta_e_norm,
        dropped_as_bot=dropped_as_bot,
        population_mean=mean,
        population_stddev=stddev
    )

    ls_result_dicts = [r.model_dump() for r in payload.result]

    stage1 = Stage1Output(
        annotation_id=payload.annotation_id,
        task_id=payload.task_id,
        wiggle_seed=wiggle_seed,
        m_initial=cache_entry.m_initial,
        m_wiggled=cache_entry.m_wiggled,
        ls_result=ls_result_dicts,
        effort=effort,
        dropped_as_bot=dropped_as_bot,
        model_version=cache_entry.model_version,
        image_width=cache_entry.image_width,
        image_height=cache_entry.image_height,
        is_honeypot=cache_entry.is_honeypot,
        m_gold=cache_entry.m_gold,
        label=cache_entry.label,
        policy_obs=cache_entry.policy_obs,
        action_z=cache_entry.action_z,
        action_log_prob=cache_entry.action_log_prob,
    )

    return stage1
