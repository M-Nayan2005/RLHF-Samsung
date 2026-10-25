"""
The learned-policy fields ride from the wiggle cache to the replay buffer, and
an annotation whose beacon never arrived is not rewarded as zero effort.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from common.schemas.label_studio_webhook import LSAnnotationUpdatedPayload
from common.schemas.redis_event import RedisEventEnvelope
from common.schemas.tier3_rlhf import WiggleCacheEntry
from tier3_worker.buffer import InMemoryReplayBuffer
from tier3_worker.stage1 import process_envelope
from tier3_worker import stage2

from .test_stage1 import dummy_ls_payload, dummy_wiggle_cache

POLICY = {
    "policy_obs": [0.1] * 8,
    "action_z": [0.5, -0.2, 0.0, 0.3, 1.1],
    "action_log_prob": -7.25,
}


def _envelope(payload: dict) -> RedisEventEnvelope:
    body = LSAnnotationUpdatedPayload(**payload)
    return RedisEventEnvelope(
        event_id="evt_1", idempotency_key=body.annotation_id,
        enqueued_at="2026-08-18T10:16:40Z", payload=body,
    )


def _db():
    pool = MagicMock()
    conn = AsyncMock()
    pool.acquire.return_value.__aenter__.return_value = conn
    conn.fetchval.side_effect = [None, "ann_9f8e7d"]
    conn.fetchrow.return_value = {"count": 2, "mean": 5.0, "m2": 2.0}
    return pool, conn


@pytest.mark.asyncio
async def test_policy_fields_reach_the_replay_buffer():
    pool, _ = _db()
    redis_pool = AsyncMock()
    redis_pool.get.return_value = WiggleCacheEntry(**dummy_wiggle_cache, **POLICY).model_dump_json()

    stage1 = await process_envelope(_envelope(dummy_ls_payload), redis_pool, pool)
    assert stage1.action_z == POLICY["action_z"]
    assert stage1.action_log_prob == POLICY["action_log_prob"]

    buffer = InMemoryReplayBuffer()
    stage2.configure(writer=buffer)
    try:
        await stage2.process_stage1(stage1)
    finally:
        await stage2.shutdown()

    [row] = buffer.rows
    assert row["policy_obs"] == POLICY["policy_obs"]
    assert row["action_z"] == POLICY["action_z"]
    assert row["action_log_prob"] == POLICY["action_log_prob"]


@pytest.mark.asyncio
async def test_missing_beacon_is_dropped_not_rewarded_as_free():
    pool, conn = _db()
    redis_pool = AsyncMock()
    redis_pool.get.return_value = WiggleCacheEntry(**dummy_wiggle_cache).model_dump_json()

    payload = dict(dummy_ls_payload)
    payload["effort_telemetry"] = {
        "click_count": 0, "cursor_path_length_px": 0.0, "dwell_time_ms": 0, "wiggle_seed": "seed_7781",
    }
    assert await process_envelope(_envelope(payload), redis_pool, pool) is None
    # The zero never entered the population statistics.
    conn.fetchrow.assert_not_called()
