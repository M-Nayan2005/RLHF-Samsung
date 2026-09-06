import asyncio
import os
import uuid
import logging
from datetime import datetime, timezone

from ppo import PPOTrainer
from deploy import BlueGreenDeployer
from dataset import ReplayBufferSampler

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
POLL_INTERVAL_SECONDS = 5
BATCH_SIZE = 64

class Tier4Poller:
    def __init__(self):
        self.current_serving_version = os.getenv("INITIAL_SERVING_VERSION", "serving-ui-stochastic-0.1.0")
        
        self.sampler = ReplayBufferSampler(DATABASE_URL)
        self.ppo_trainer = PPOTrainer()
        self.deployer = BlueGreenDeployer(models_dir=MODELS_DIR, redis_url=REDIS_URL)

    async def poll_once(self):
        # Sample randomized batch
        tuples = await self.sampler.sample_random_batch(self.current_serving_version, BATCH_SIZE)
        
        if len(tuples) < BATCH_SIZE:
            return

        logger.info(f"Sampled exactly {BATCH_SIZE} random tuples! Initiating PPO Training run...")
        
        batch_id = str(uuid.uuid4())
        tuple_ids = [t['tuple_id'] for t in tuples]
        
        # Mark them consumed immediately
        await self.sampler.mark_consumed(tuple_ids, batch_id)
        logger.info(f"Flipped consumed_by_ppo=TRUE for batch {batch_id}.")
        
        started_at = datetime.now(timezone.utc)
        
        try:
            # Run the PPO Actor-Critic step
            metrics = self.ppo_trainer.train_step(tuples)
            
            # Hot-Swap Deploy
            new_version = self.deployer.deploy_weights(self.current_serving_version)
            
            # Write audit trail to tier3.ppo_training_runs
            completed_at = datetime.now(timezone.utc)
            await self.sampler.conn.execute("""
                INSERT INTO tier3.ppo_training_runs 
                (batch_id, model_version_in, model_version_out, batch_size, tuple_ids, 
                 actor_loss, critic_loss, mean_reward, started_at, completed_at, status, deployed_via)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
            """, 
            uuid.UUID(batch_id), self.current_serving_version, new_version, BATCH_SIZE, [uuid.UUID(tid) for tid in tuple_ids],
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
            await self.sampler.conn.execute("""
                INSERT INTO tier3.ppo_training_runs 
                (batch_id, model_version_in, batch_size, tuple_ids, started_at, completed_at, status, error_message)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """, 
            uuid.UUID(batch_id), self.current_serving_version, BATCH_SIZE, [uuid.UUID(tid) for tid in tuple_ids],
            started_at, completed_at, "failed", str(e))
            logger.info("Audited FAILED PPO training run. Tuples remain consumed.")

    async def run_forever(self):
        await self.sampler.connect()
        logger.info(f"Started Poller. Tracking model_version: {self.current_serving_version}")
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
