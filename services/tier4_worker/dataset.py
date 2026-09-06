import asyncpg
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ReplayBufferSampler:
    """
    Queries tier3.replay_buffer and samples randomized training batches of size 64.
    """
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.conn = None

    async def connect(self):
        if not self.conn:
            self.conn = await asyncpg.connect(self.db_url)

    async def sample_random_batch(self, model_version: str, batch_size: int = 64) -> List[Dict[str, Any]]:
        """
        Samples a randomized training batch of size `batch_size`.
        """
        await self.connect()
        async with self.conn.transaction():
            rows = await self.conn.fetch("""
                SELECT tuple_id, state_s_t, action_a_t, reward_r_t
                FROM tier3.replay_buffer
                WHERE consumed_by_ppo = FALSE 
                  AND model_version = $1
                ORDER BY random() 
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            """, model_version, batch_size)
            
            if len(rows) < batch_size:
                logger.info(f"Not enough unconsumed tuples for {model_version}. Found {len(rows)}, need {batch_size}.")
                return []
                
            return [dict(row) for row in rows]

    async def mark_consumed(self, tuple_ids: List[str], batch_id: str):
        """Marks a batch of tuples as consumed by PPO."""
        await self.connect()
        await self.conn.execute("""
            UPDATE tier3.replay_buffer
            SET consumed_by_ppo = TRUE,
                consumed_at = now(),
                ppo_batch_id = $2
            WHERE tuple_id = ANY($1)
        """, tuple_ids, batch_id)
