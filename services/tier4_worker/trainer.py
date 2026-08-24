import asyncio
import asyncpg
import os
import uuid
import logging
import json
from datetime import datetime, timezone
from ppo import PPOTrainerStub
from deploy import BlueGreenDeployer

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
POLL_INTERVAL_SECONDS = 5
BATCH_SIZE = 64

class Tier4Poller:
    def __init__(self):
        self.current_serving_version = os.getenv("INITIAL_SERVING_VERSION", "serving-ui-stochastic-0.1.0")
        self.ppo_trainer = PPOTrainerStub()
        self.deployer = BlueGreenDeployer()
        self.conn = None

    async def connect(self):
        self.conn = await asyncpg.connect(DATABASE_URL)
        logger.info(f"Connected to database. Tracking model_version: {self.current_serving_version}")

    async def poll_once(self):
        # We need exactly BATCH_SIZE (64) rows matching the current version
        # We wrap the claim in a transaction
        async with self.conn.transaction():
            # Query for the oldest unconsumed rows
            rows = await self.conn.fetch("""
                SELECT tuple_id, state_s_t, action_a_t, reward_r_t
                FROM tier3.replay_buffer
                WHERE consumed_by_ppo = FALSE 
                  AND model_version = $1
                ORDER BY created_at ASC 
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            """, self.current_serving_version, BATCH_SIZE)

            if len(rows) < BATCH_SIZE:
                logger.info(f"Only found {len(rows)}/{BATCH_SIZE} tuples for version '{self.current_serving_version}'. Waiting...")
                return

            logger.info(f"Found exactly {BATCH_SIZE} tuples! Initiating PPO Training run...")
            
            # We have a full batch. Mark them consumed immediately.
            batch_id = uuid.uuid4()
            tuple_ids = [row['tuple_id'] for row in rows]
            now_tz = datetime.now(timezone.utc)
            
            await self.conn.execute("""
                UPDATE tier3.replay_buffer
                SET consumed_by_ppo = TRUE,
                    consumed_at = $2,
                    ppo_batch_id = $3
                WHERE tuple_id = ANY($1)
            """, tuple_ids, now_tz, batch_id)
            
            logger.info(f"Flipped consumed_by_ppo=TRUE for batch {batch_id}.")
            
            # Convert rows to dicts for the trainer
            tuples = [dict(row) for row in rows]
            
            # Start tracking the training run
            started_at = datetime.now(timezone.utc)
            
            try:
                # Run the PPO Actor-Critic step
                metrics = self.ppo_trainer.train_step(tuples)
                
                # Mock Deploy
                new_version = self.deployer.deploy_weights(self.current_serving_version)
                
                # Write audit trail to tier3.ppo_training_runs
                completed_at = datetime.now(timezone.utc)
                await self.conn.execute("""
                    INSERT INTO tier3.ppo_training_runs 
                    (batch_id, model_version_in, model_version_out, batch_size, tuple_ids, 
                     actor_loss, critic_loss, mean_reward, started_at, completed_at, status, deployed_via)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, 
                batch_id, self.current_serving_version, new_version, BATCH_SIZE, tuple_ids,
                metrics["actor_loss"], metrics["critic_loss"], metrics["mean_reward"],
                started_at, completed_at, "succeeded", self.deployer.deployment_target)
                
                logger.info("Successfully audited PPO training run.")
                
                # Update our local pointer so the next poll cycle only picks up tuples for the NEW model
                self.current_serving_version = new_version
                
            except Exception as e:
                logger.error(f"Training failed: {e}")
                completed_at = datetime.now(timezone.utc)
                # We do NOT roll back the consumed_by_ppo flip! 
                # We just write the failure to the audit table.
                await self.conn.execute("""
                    INSERT INTO tier3.ppo_training_runs 
                    (batch_id, model_version_in, batch_size, tuple_ids, started_at, completed_at, status, error_message)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, 
                batch_id, self.current_serving_version, BATCH_SIZE, tuple_ids,
                started_at, completed_at, "failed", str(e))
                logger.info("Audited FAILED PPO training run. Tuples remain consumed.")

    async def run_forever(self):
        await self.connect()
        while True:
            try:
                await self.poll_once()
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
            
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

if __name__ == "__main__":
    poller = Tier4Poller()
    logger.info("Starting Tier 4 PPO Rollout Poller...")
    asyncio.run(poller.run_forever())
