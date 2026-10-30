"""
Pull on-policy training batches out of `tier3.replay_buffer`.

Despite the table's inherited name, this is not experience replay
(`docs/authority.md` C1 rejects that for PPO): a batch is the oldest N
unconsumed tuples of exactly one `model_version`, each consumed once. Tuples
from a superseded checkpoint are never trained on; they stay unconsumed as an
audit record of what the old policy was shown to do.
"""
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import asyncpg

logger = logging.getLogger(__name__)

_SCHEMA_CANDIDATES = (
    Path("/app/infra/sql/tier3_schema.sql"),
    Path(__file__).resolve().parents[2] / "infra" / "sql" / "tier3_schema.sql",
)

_CLAIM_SQL = """
SELECT tuple_id, policy_obs, action_z, action_log_prob, reward_r_t, label, is_honeypot
FROM tier3.replay_buffer
WHERE consumed_by_ppo = FALSE
  AND model_version = $1
  AND action_z IS NOT NULL
  AND policy_obs IS NOT NULL
  AND action_log_prob IS NOT NULL
ORDER BY created_at
LIMIT $2
FOR UPDATE SKIP LOCKED
"""


def normalize_dsn(dsn: str) -> str:
    """`postgresql+asyncpg://` (SQLAlchemy spelling, used in .env.example) -> `postgresql://`."""
    scheme, sep, rest = dsn.partition("://")
    if sep and "+" in scheme:
        return f"{scheme.split('+', 1)[0]}://{rest}"
    return dsn


def _json(value: Any) -> Any:
    # asyncpg hands jsonb back as text unless a codec is registered.
    return json.loads(value) if isinstance(value, str) else value


class ReplayBufferSampler:
    def __init__(self, db_url: str):
        self.db_url = normalize_dsn(db_url or "")
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        if self.pool is None:
            self.pool = await asyncpg.create_pool(dsn=self.db_url, min_size=1, max_size=4)

    async def close(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def ensure_schema(self) -> None:
        """Apply the shared tier3 DDL. Idempotent; lets Tier 4 start before Tier 3 has written anything."""
        path = next((p for p in _SCHEMA_CANDIDATES if p.is_file()), None)
        if path is None:
            logger.warning("tier3_schema.sql not found; assuming the schema is already applied")
            return
        await self.connect()
        async with self.pool.acquire() as conn:
            await conn.execute(path.read_text(encoding="utf-8"))

    async def claim_batch(self, model_version: str, batch_size: int, batch_id: str) -> List[Dict[str, Any]]:
        """
        Atomically claim the oldest `batch_size` trainable tuples of `model_version`.

        All or nothing: below `batch_size` nothing is claimed, so a partial batch
        keeps accumulating instead of being trained on early.
        """
        await self.connect()
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                rows = await conn.fetch(_CLAIM_SQL, model_version, batch_size)
                if len(rows) < batch_size:
                    return []
                ids = [row["tuple_id"] for row in rows]
                await conn.execute(
                    """
                    UPDATE tier3.replay_buffer
                    SET consumed_by_ppo = TRUE, consumed_at = now(), ppo_batch_id = $2
                    WHERE tuple_id = ANY($1::uuid[])
                    """,
                    ids, uuid.UUID(batch_id),
                )
        return [
            {
                "tuple_id": str(row["tuple_id"]),
                "policy_obs": _json(row["policy_obs"]),
                "action_z": _json(row["action_z"]),
                "action_log_prob": row["action_log_prob"],
                "reward_r_t": row["reward_r_t"],
                "label": row["label"],
                "is_honeypot": row["is_honeypot"],
            }
            for row in rows
        ]

    async def pending_count(self, model_version: str) -> int:
        await self.connect()
        async with self.pool.acquire() as conn:
            return await conn.fetchval(
                """
                SELECT count(*) FROM tier3.replay_buffer
                WHERE consumed_by_ppo = FALSE AND model_version = $1 AND action_z IS NOT NULL
                """,
                model_version,
            )

    async def record_run(self, **fields: Any) -> None:
        """Write one row of `tier3.ppo_training_runs`."""
        await self.connect()
        columns = list(fields)
        placeholders = ", ".join(f"${i}" for i in range(1, len(columns) + 1))
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO tier3.ppo_training_runs ({', '.join(columns)}) VALUES ({placeholders})",
                *fields.values(),
            )
