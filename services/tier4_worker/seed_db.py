import asyncio
import asyncpg
import json
import uuid
import os
from datetime import datetime, timezone

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/postgres")

async def seed():
    conn = await asyncpg.connect(DATABASE_URL)
    
    # Read the real DDL file
    with open("infra/sql/tier3_schema.sql", "r") as f:
        ddl = f.read()
    
    await conn.execute(ddl)
    print("Ensured tier3 schema and tables exist using real DDL.")
    
    # Insert exactly 65 rows (so 1 is left over after a 64-batch poll)
    model_version = "serving-ui-stochastic-0.1.0"
    
    tuples_inserted = 0
    for i in range(65):
        tuple_id = uuid.uuid4()
        wiggle_seed = f"seed_{i}"
        
        await conn.execute("""
            INSERT INTO tier3.replay_buffer 
            (tuple_id, wiggle_seed, task_id, annotation_id, state_s_t, action_a_t, reward_r_t, 
             delta_iou, delta_e_norm, alpha, beta, model_version, created_at, consumed_by_ppo)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (annotation_id) DO NOTHING
        """,
        tuple_id,
        wiggle_seed,
        f"task_{i}",
        f"ann_{i}",
        json.dumps({"points": []}), # Mock state_s_t (M_initial)
        json.dumps({"points": []}), # Mock action_a_t (M_wiggled)
        0.5 + (i * 0.01),           # Fake reward
        0.1,                        # delta_iou
        0.2,                        # delta_e_norm
        1.0,                        # alpha
        0.3,                        # beta
        model_version,
        datetime.now(timezone.utc),
        False
        )
        tuples_inserted += 1
        
    print(f"Attempted to insert {tuples_inserted} mock rows into tier3.replay_buffer for version '{model_version}'.")
    await conn.close()

if __name__ == "__main__":
    asyncio.run(seed())
