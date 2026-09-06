"""
Dev 2's entrypoint — the second half of the Tier 3 worker pipeline.

    Dev 1  process_envelope(envelope) -> Optional[Stage1Output]
    Dev 2  process_stage1(stage1)     -> None

Dev 1 calls this in-process, in the same worker, as a direct function call
rather than through a second queue. Everything Dev 2 needs arrives on the
`Stage1Output`; the raw envelope and the raw telemetry are never touched here.

Nodes implemented, in order:

    Geometric Delta Engine        -> reward.compute_geometric_delta
    E-DRDE Scalar Evaluator       -> reward.evaluate_edrde
    State-Action-Reward Aggregator-> aggregator.build_experience_tuple
    Offline Replay Buffer write   -> buffer.PostgresReplayBuffer

**What is dropped versus what is raised.** These are different failures and
conflating them is how a pipeline either dies on one bad polygon or silently
discards a day of training data.

  - *Data-shaped* problems — bot-flagged effort, a missing `model_version`, a
    degenerate or unrepairable polygon, an annotation with no polygon regions —
    are logged at WARNING/ERROR and dropped. `process_stage1` returns normally
    so Dev 1's loop keeps consuming. One unusable annotation must never stop
    the queue.
  - *Infrastructure* problems — Postgres unreachable, pool exhausted — are
    logged and re-raised, because they are not properties of this tuple and
    will apply equally to the next ten thousand. Dev 1's consumer decides
    whether to back off, retry, or park the message; swallowing them here would
    turn an outage into invisible data loss. Set
    `TIER3_RAISE_ON_WRITE_FAILURE=false` to invert that if the operational
    preference turns out to be the opposite.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from pydantic import ValidationError

from common.schemas.tier3_rlhf import Stage1Output

from .aggregator import build_experience_tuple
from .buffer import PostgresReplayBuffer, ReplayBufferWriter
from .config import Settings, load_settings, log_provisional_hyperparameters
from .geometry import GeometryError
from .reward import compute_geometric_delta, evaluate_edrde

log = logging.getLogger(__name__)

_settings: Optional[Settings] = None
_writer: Optional[ReplayBufferWriter] = None


# ---------------------------------------------------------------------------
# Wiring. Kept as module state so `process_stage1` can hold the exact signature
# the handoff contract specifies while still being injectable from tests.
# ---------------------------------------------------------------------------

def configure(settings: Optional[Settings] = None, writer: Optional[ReplayBufferWriter] = None) -> None:
    """
    Set up the stage. Call once at worker startup, before the first envelope.

    Dev 1: `configure()` with no arguments is the production path — it reads the
    environment and builds a Postgres-backed writer. Call `await
    shutdown()` on the way out so the pool closes cleanly.
    """
    global _settings, _writer
    _settings = settings or load_settings()
    log_provisional_hyperparameters(_settings)
    _writer = writer if writer is not None else PostgresReplayBuffer(_settings)


async def shutdown() -> None:
    global _writer
    close = getattr(_writer, "close", None)
    if close is not None:
        await close()
    _writer = None


def get_settings() -> Settings:
    if _settings is None:
        configure()
    assert _settings is not None
    return _settings


def get_writer() -> ReplayBufferWriter:
    if _writer is None:
        configure()
    assert _writer is not None
    return _writer


# ---------------------------------------------------------------------------
# The entrypoint
# ---------------------------------------------------------------------------

async def process_stage1(stage1: Stage1Output) -> None:
    """
    Turn one accepted annotation into one `tier3.replay_buffer` row.

    Returns None in every case, including every drop. Raises only when the
    write itself fails for infrastructural reasons — see the module docstring.
    """
    settings = get_settings()

    if stage1.dropped_as_bot or stage1.effort.dropped_as_bot:
        log.warning(
            "DROP task=%s annotation=%s reason=bot_flagged "
            "(delta_e_raw=%s) — excluded from the replay buffer",
            stage1.task_id, stage1.annotation_id, stage1.effort.delta_e_raw,
        )
        return

    if settings.require_model_version and not stage1.model_version:
        log.error(
            "DROP task=%s annotation=%s reason=missing_model_version — "
            "tier3.replay_buffer.model_version is NOT NULL and Tier 4 batches on it, "
            "so this tuple is unstorable and untrainable. Populate "
            "WiggleCacheEntry.model_version in the Tier 2 cache writer.",
            stage1.task_id, stage1.annotation_id,
        )
        return

    try:
        delta, mode = compute_geometric_delta(stage1, settings)
    except GeometryError as exc:
        log.error(
            "DROP task=%s annotation=%s reason=bad_geometry (%s) — a reward computed "
            "from a degenerate polygon is worse than none, because it looks plausible",
            stage1.task_id, stage1.annotation_id, exc,
        )
        return

    try:
        reward = evaluate_edrde(stage1, delta, settings)
        experience = build_experience_tuple(stage1, reward)
    except (ValidationError, ValueError) as exc:
        log.error(
            "DROP task=%s annotation=%s reason=invalid_tuple (%s)",
            stage1.task_id, stage1.annotation_id, exc,
        )
        return

    try:
        inserted = await get_writer().write(
            experience,
            delta,
            reward,
            label=stage1.label,
            is_honeypot=stage1.is_honeypot,
        )
    except Exception:
        log.exception(
            "replay buffer write FAILED for task=%s annotation=%s — this is an "
            "infrastructure failure, not a property of this tuple, so it is being "
            "re-raised rather than swallowed into silent data loss",
            stage1.task_id, stage1.annotation_id,
        )
        if _raise_on_write_failure():
            raise
        return

    if not inserted:
        log.info(
            "duplicate task=%s annotation=%s — already in the replay buffer, "
            "ON CONFLICT swallowed the insert",
            stage1.task_id, stage1.annotation_id,
        )
        return

    log.info(
        "reward task=%s annotation=%s mode=%s delta_iou=%.6f delta_e_norm=%.6f "
        "r_t=%.6f model_version=%s tuple=%s",
        stage1.task_id, stage1.annotation_id, mode,
        delta.delta_iou, reward.delta_e_norm, reward.r_t,
        experience.model_version, experience.tuple_id,
    )


def _raise_on_write_failure() -> bool:
    return os.environ.get("TIER3_RAISE_ON_WRITE_FAILURE", "true").strip().lower() in {
        "1", "true", "yes", "on",
    }
