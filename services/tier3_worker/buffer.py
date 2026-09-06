"""
Writes into `tier3.replay_buffer`.

All Postgres access is `asyncpg` — the acceptance criteria forbid a synchronous
driver anywhere near the event loop, and `services/routing_qa` has already been
patched once for exactly that (`4ac83d4 fix(routing_qa): unblock event loop`).

**Idempotency is enforced twice on purpose.** Dev 1's `tier3
.processed_annotations` ledger is the first line: a redelivered webhook is
recognised and dropped before it reaches the reward math. This writer is the
second: `INSERT ... ON CONFLICT (annotation_id) DO NOTHING` against the UNIQUE
constraint. The two run in separate transactions, so two workers racing the
same redelivery can both clear the ledger check; only the constraint can stop
the duplicate row. Acceptance criterion "replay the same annotation_id twice,
confirm only one replay_buffer row" is satisfied by the constraint alone, which
is what makes it worth having.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Protocol

from common.schemas.tier3_rlhf import EDRDEReward, ExperienceTuple, GeometricDelta

from .config import Settings

log = logging.getLogger(__name__)

# Where the DDL lives, in the container and on a laptop respectively.
_SCHEMA_CANDIDATES = (
    Path("/app/infra/sql/tier3_schema.sql"),
    Path(__file__).resolve().parents[2] / "infra" / "sql" / "tier3_schema.sql",
)


def _schema_path() -> Optional[Path]:
    return next((p for p in _SCHEMA_CANDIDATES if p.is_file()), None)


def as_timestamptz(value: Any) -> datetime:
    """
    Coerce an ISO-8601 string into the `datetime` asyncpg demands.

    `ExperienceTuple.created_at` is a string, because every timestamp in the
    frozen contracts is. asyncpg does not accept a string for a `timestamptz`
    parameter the way psycopg2 does — it encodes arguments client-side and
    raises `DataError: expected a datetime.date or datetime.datetime instance`.
    A `$n::timestamptz` cast does not rescue it either: the cast is server-side
    and the bind never gets that far. So the conversion has to happen here.

    A naive timestamp is read as UTC. Every producer in this codebase emits UTC
    with a trailing `Z`, and silently attaching the worker's local zone to a
    timestamp that is actually UTC would skew `created_at` ordering, which is
    exactly what Tier 4 batches on.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"created_at is not an ISO-8601 timestamp: {value!r}") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


_INSERT_SQL = """
INSERT INTO tier3.replay_buffer (
    tuple_id, wiggle_seed, task_id, annotation_id,
    state_s_t, action_a_t, reward_r_t,
    delta_iou, delta_e_norm, alpha, beta,
    model_version, created_at, consumed_by_ppo,
    label, is_honeypot
) VALUES (
    $1::uuid, $2, $3, $4,
    $5::jsonb, $6::jsonb, $7,
    $8, $9, $10, $11,
    $12, $13::timestamptz, FALSE,
    $14, $15
)
ON CONFLICT (annotation_id) DO NOTHING
RETURNING tuple_id
"""


class ReplayBufferWriter(Protocol):
    """What `stage2` needs from a buffer. Lets the tests substitute an in-memory double."""

    async def write(
        self,
        tuple_: ExperienceTuple,
        delta: GeometricDelta,
        reward: EDRDEReward,
        label: Optional[str],
        is_honeypot: bool,
    ) -> bool:
        """True if a row was inserted, False if the annotation_id was already present."""
        ...


class PostgresReplayBuffer:
    """asyncpg-backed writer. Owns its own pool so Dev 1's consumer can share one process."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: Any = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        import asyncpg  # imported lazily so the reward math is testable without a driver

        self._pool = await asyncpg.create_pool(
            dsn=self._settings.pg_dsn,
            min_size=1,
            max_size=8,
            command_timeout=self._settings.db_statement_timeout_ms / 1000,
        )
        if self._settings.apply_schema_on_start:
            await self.ensure_schema()
        log.info("tier3 replay buffer pool ready")

    async def ensure_schema(self) -> None:
        """
        Apply `infra/sql/tier3_schema.sql`.

        Following `services/routing_qa/db.py`, which also creates its own tables
        on startup: the compose `postgres` service mounts no init directory and
        keeps a named volume, so a `docker-entrypoint-initdb.d` script would run
        only against a brand-new volume and silently skip every existing
        deployment. Applying from the worker is the option that actually
        converges. Every statement in the file is IF NOT EXISTS / ON CONFLICT,
        so this is a no-op on an already-migrated database.
        """
        path = _schema_path()
        if path is None:
            log.warning(
                "tier3_schema.sql not found; assuming the schema is already applied. "
                "If the first write fails on a missing relation, apply it by hand: "
                "psql \"$DATABASE_URL\" -f infra/sql/tier3_schema.sql"
            )
            return
        async with self._pool.acquire() as conn:
            await conn.execute(path.read_text(encoding="utf-8"))
        log.info("tier3 schema applied from %s", path)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def write(
        self,
        tuple_: ExperienceTuple,
        delta: GeometricDelta,
        reward: EDRDEReward,
        label: Optional[str],
        is_honeypot: bool,
    ) -> bool:
        if self._pool is None:
            await self.connect()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                _INSERT_SQL,
                tuple_.tuple_id,
                tuple_.wiggle_seed,
                tuple_.task_id,
                tuple_.annotation_id,
                json.dumps(tuple_.state_s_t.model_dump()),
                json.dumps(tuple_.action_a_t.model_dump()),
                tuple_.reward_r_t,
                delta.delta_iou,
                reward.delta_e_norm,
                reward.alpha,
                reward.beta,
                tuple_.model_version,
                as_timestamptz(tuple_.created_at),
                label,
                is_honeypot,
            )

        # `RETURNING` yields no row when ON CONFLICT swallowed the insert.
        return row is not None


class InMemoryReplayBuffer:
    """
    Test/dev double. Also what `MOCK_MODE` uses so the pipeline can be exercised
    end to end without Postgres, the same way Track 3 was built and verified.
    """

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self._seen: set[str] = set()

    async def write(
        self,
        tuple_: ExperienceTuple,
        delta: GeometricDelta,
        reward: EDRDEReward,
        label: Optional[str],
        is_honeypot: bool,
    ) -> bool:
        if tuple_.annotation_id in self._seen:
            return False
        self._seen.add(tuple_.annotation_id)
        self.rows.append(
            {
                "tuple_id": tuple_.tuple_id,
                "wiggle_seed": tuple_.wiggle_seed,
                "task_id": tuple_.task_id,
                "annotation_id": tuple_.annotation_id,
                "state_s_t": tuple_.state_s_t.model_dump(),
                "action_a_t": tuple_.action_a_t.model_dump(),
                "reward_r_t": tuple_.reward_r_t,
                "delta_iou": delta.delta_iou,
                "delta_e_norm": reward.delta_e_norm,
                "alpha": reward.alpha,
                "beta": reward.beta,
                "model_version": tuple_.model_version,
                "created_at": tuple_.created_at,
                "consumed_by_ppo": False,
                "label": label,
                "is_honeypot": is_honeypot,
            }
        )
        return True
