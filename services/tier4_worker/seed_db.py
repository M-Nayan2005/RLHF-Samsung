"""
Standalone Tier 4 smoke test: insert synthetic trainable tuples for the serving checkpoint.

    docker compose run --rm tier4_worker python -m tier4_worker.seed_db --count 65

Each tuple is a real draw from the current policy (so its log-prob is genuine
and the trainer's on-policy check passes), with a made-up reward that peaks at
normal_offset = +1. A few batches should move the policy's mean toward it.
Clearly not human data - use scripts/simulate_annotators.py or Label Studio
for the real loop.
"""
import argparse
import asyncio
import json
import os
import uuid

import asyncpg
import numpy as np

from common import rl_policy

from .dataset import ReplayBufferSampler, normalize_dsn


async def seed(count: int) -> None:
    sampler = ReplayBufferSampler(os.getenv("DATABASE_URL", ""))
    await sampler.ensure_schema()
    await sampler.close()

    store = rl_policy.PolicyStore(
        os.getenv("MODELS_DIR", "/app/models"),
        os.getenv("SERVING_UI_MODEL_VERSION", "serving-ui-stochastic-0.1.0"),
    )
    policy = store.bootstrap() if not store.pointer.exists() else store.current()
    rng = np.random.default_rng()

    conn = await asyncpg.connect(normalize_dsn(os.getenv("DATABASE_URL", "")))
    try:
        for i in range(count):
            square = [[100, 100], [100 + 50 + i % 7 * 10, 100], [150 + i % 7 * 10, 200], [100, 200]]
            obs = rl_policy.observation(square, 640, 480, 0.02, 0.1)
            z = policy.mean(obs) + policy.std() * rng.standard_normal(5)
            reward = 1.0 - (z[4] - 1.0) ** 2
            await conn.execute(
                """
                INSERT INTO tier3.replay_buffer
                (tuple_id, wiggle_seed, task_id, annotation_id, state_s_t, action_a_t, reward_r_t,
                 delta_iou, delta_e_norm, alpha, beta, model_version, label,
                 policy_obs, action_z, action_log_prob)
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7, 0, 0, 1.0, 0.3, $8, 'synthetic',
                        $9::jsonb, $10::jsonb, $11)
                """,
                uuid.uuid4(), f"synthetic_{uuid.uuid4().hex[:8]}", f"task_syn_{i}", f"ann_syn_{uuid.uuid4().hex}",
                json.dumps({"points": square}), json.dumps({"points": square}), float(reward),
                policy.version, json.dumps(obs), json.dumps(z.tolist()), policy.log_prob(obs, z),
            )
    finally:
        await conn.close()
    print(f"Inserted {count} synthetic tuples for {policy.version}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=65)
    asyncio.run(seed(parser.parse_args().count))
