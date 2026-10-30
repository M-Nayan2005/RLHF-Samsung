"""
Tier 4 worker: wait for a full on-policy batch, run PPO, publish the new policy.

    python -m tier4_worker.trainer

Loop:
  1. Read the serving checkpoint (`<models_dir>/policy/current.json`).
  2. Claim the oldest BATCH_SIZE tuples Tier 3 wrote for exactly that version.
  3. PPO-update the policy and its critic on them.
  4. Publish the result as the next version; serving picks it up on its next request.
  5. Record the run in `tier3.ppo_training_runs`.

Tuples served under the old version that arrive after step 4 are never trained
on - PPO is on-policy (`docs/authority.md` C1).
"""
import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone

from common import rl_policy

from .dataset import ReplayBufferSampler
from .deploy import PolicyPublisher
from .ppo import PPOConfig, PPOTrainer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("tier4_trainer")

DATABASE_URL = os.getenv("DATABASE_URL", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MODELS_DIR = os.getenv("MODELS_DIR", "/app/models")
POLL_INTERVAL_SECONDS = float(os.getenv("TIER4_POLL_INTERVAL_S", "5"))
# 64 is the batch size the Tier 3/4 plan specifies.
BATCH_SIZE = int(os.getenv("TIER4_BATCH_SIZE", "64"))
INITIAL_VERSION = os.getenv("SERVING_UI_MODEL_VERSION", "serving-ui-stochastic-0.1.0")


class Tier4Poller:
    def __init__(self):
        self.sampler = ReplayBufferSampler(DATABASE_URL)
        self.publisher = PolicyPublisher(MODELS_DIR, INITIAL_VERSION, REDIS_URL)
        self.ppo_trainer = PPOTrainer(PPOConfig.from_env())
        self._last_pending = None

    async def poll_once(self) -> bool:
        """One cycle. True if a batch was trained."""
        current = self.publisher.store.current()
        batch_id = str(uuid.uuid4())
        tuples = await self.sampler.claim_batch(current.version, BATCH_SIZE, batch_id)

        if not tuples:
            pending = await self.sampler.pending_count(current.version)
            if pending != self._last_pending:
                logger.info("Waiting for a batch: %d/%d trainable tuples for %s", pending, BATCH_SIZE, current.version)
                self._last_pending = pending
            return False
        self._last_pending = None

        logger.info("Claimed batch %s: %d tuples of %s. Running PPO...", batch_id, len(tuples), current.version)
        started_at = datetime.now(timezone.utc)
        tuple_ids = [uuid.UUID(t["tuple_id"]) for t in tuples]

        try:
            actor, critic, metrics = self.ppo_trainer.train(
                current, tuples, critic_state=self.publisher.load_critic(current.version)
            )
            new_params = actor.export(rl_policy.next_version(current.version), parent_version=current.version)
            self.publisher.publish(new_params, critic.state_dict())

            await self.sampler.record_run(
                batch_id=uuid.UUID(batch_id),
                model_version_in=current.version,
                model_version_out=new_params.version,
                batch_size=len(tuples),
                tuple_ids=tuple_ids,
                actor_loss=metrics["actor_loss"],
                critic_loss=metrics["critic_loss"],
                mean_reward=metrics["mean_reward"],
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="succeeded",
                deployed_via=f"policy_pointer:{self.publisher.store.pointer}",
            )
            logger.info(
                "Batch %s: %s -> %s. mean_reward=%.4f kl=%.4f mean_normal_offset=%.3f std_normal_offset=%.3f",
                batch_id, current.version, new_params.version, metrics["mean_reward"],
                metrics["approx_kl"], metrics["mean_normal_offset"], metrics["std_normal_offset"],
            )
            return True

        except Exception as e:
            # The tuples stay consumed: retrying a batch that failed for a data
            # reason would fail forever, and on-policy data cannot be re-served.
            logger.exception("Training batch %s failed: %s", batch_id, e)
            await self.sampler.record_run(
                batch_id=uuid.UUID(batch_id),
                model_version_in=current.version,
                batch_size=len(tuples),
                tuple_ids=tuple_ids,
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
                status="failed",
                error_message=str(e)[:2000],
            )
            return False

    async def run_forever(self):
        await self.sampler.ensure_schema()
        current = self.publisher.store.bootstrap()
        logger.info("Tier 4 trainer up. Serving checkpoint: %s; batch size %d", current.version, BATCH_SIZE)
        while True:
            try:
                trained = await self.poll_once()
            except Exception as e:
                logger.error(f"Error in polling loop: {e}")
                trained = False
            if not trained:
                await asyncio.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    asyncio.run(Tier4Poller().run_forever())
