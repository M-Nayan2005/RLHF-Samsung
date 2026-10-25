"""
The Tier 2 -> Tier 3 bridge: `wiggle_cache:{wiggle_seed}` in Redis.

Tier 3 joins every annotation back to what was served through this key
(`common/schemas/tier3_rlhf.py::WiggleCacheEntry`). Without it Tier 3 has no
`M_initial`, no `M_wiggled` and no image dimensions, so it drops the
annotation and nothing is ever rewarded. The entry is a projection of the
`ServedWiggleRecord` this service already persists; nothing new is computed.

A failed write never fails the serve. The annotator still gets a mask; that
one rollout just produces no reward, and the error says so.
"""
from __future__ import annotations

import logging
from typing import Optional

from common.schemas.tier1_ingestion import PolygonMask
from common.schemas.tier3_rlhf import WiggleCacheEntry

from .config import Settings
from .models import ServedWiggleRecord

log = logging.getLogger(__name__)

KEY_PREFIX = "wiggle_cache:"


def cache_key(wiggle_seed: str) -> str:
    return f"{KEY_PREFIX}{wiggle_seed}"


def build_entry(
    record: ServedWiggleRecord,
    baseline_points,
    label: Optional[str],
) -> WiggleCacheEntry:
    """
    `baseline_points` is the task's own `baseline_mask.points`, not the
    canonicalised copy on the record, so `m_initial` is exactly Tier 1's mask.
    `m_gold` stays None: routing_qa strips the honeypot gold mask before it
    reaches this service, and Tier 3 falls back to the production form of
    delta-IoU without it.
    """
    return WiggleCacheEntry(
        wiggle_seed=record.wiggle_seed,
        task_id=record.task_id,
        image_id=record.image_id,
        m_initial=PolygonMask(points=[list(p) for p in baseline_points]),
        m_wiggled=PolygonMask(points=record.wiggled_points),
        served_at=record.served_at,
        image_width=record.image_width,
        image_height=record.image_height,
        model_version=record.model_version,
        is_honeypot=record.is_honeypot,
        m_gold=None,
        label=label,
        policy_obs=record.policy_obs,
        action_z=record.action_z,
        action_log_prob=record.action_log_prob,
    )


class WiggleCacheWriter:
    def __init__(self, settings: Settings, client=None) -> None:
        self._ttl = settings.wiggle_cache_ttl_s
        self._client = client
        self.enabled = client is not None or bool(settings.redis_url)
        if client is None and settings.redis_url:
            import redis  # lazily, so tests without a Redis driver still import this module

            self._client = redis.Redis.from_url(settings.redis_url, socket_timeout=2.0)

    def write(self, entry: WiggleCacheEntry) -> bool:
        if not self.enabled:
            return False
        try:
            self._client.set(cache_key(entry.wiggle_seed), entry.model_dump_json(), ex=self._ttl)
            return True
        except Exception as exc:  # noqa: BLE001 - any Redis failure degrades to "no rollout"
            log.error(
                "Could not write %s to Redis (%s). The mask is still served, but Tier 3 "
                "cannot join this annotation, so it produces no reward.",
                cache_key(entry.wiggle_seed), exc,
            )
            return False

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close is not None:
            close()
