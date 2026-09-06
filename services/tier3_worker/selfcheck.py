"""
Live self-check for Dev 2's half of the Tier 3 worker.

    python -m tier3_worker.selfcheck

Everything the unit suite covers, it covers offline. This covers the part it
cannot: that the DDL actually applies, that `ON CONFLICT (annotation_id)`
actually stops a duplicate at the database rather than only in the in-memory
double, and that asyncpg accepts the DSN as configured. Those are the three
things that have historically only failed when something was really running.

Idempotent and safe against a live database: the DDL is all IF NOT EXISTS, and
the probe rows it inserts are deleted again on the way out.

Exit code 0 means every check passed.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from pathlib import Path
from typing import List

from common.schemas.tier1_ingestion import PolygonMask
from common.schemas.tier3_rlhf import NormalizedEffortScore, Stage1Output

from . import stage2
from .buffer import PostgresReplayBuffer
from .config import load_settings, log_provisional_hyperparameters

log = logging.getLogger("tier3.selfcheck")

SCHEMA_PATHS = (
    Path("/app/infra/sql/tier3_schema.sql"),
    Path(__file__).resolve().parents[2] / "infra" / "sql" / "tier3_schema.sql",
)

IMAGE_WIDTH, IMAGE_HEIGHT = 800, 400


def _rect(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _probe_stage1(annotation_id: str, task_id: str) -> Stage1Output:
    """The same fixture geometry the unit suite uses, so the numbers are comparable."""
    final_percent = [
        [x / IMAGE_WIDTH * 100.0, y / IMAGE_HEIGHT * 100.0]
        for x, y in _rect(100, 50, 400, 300)
    ]
    return Stage1Output(
        annotation_id=annotation_id,
        task_id=task_id,
        wiggle_seed="selfcheck-seed",
        m_initial=PolygonMask(points=_rect(100, 50, 400, 300)),
        m_wiggled=PolygonMask(points=_rect(120, 60, 420, 310)),
        ls_result=[
            {
                "id": "region_1",
                "type": "polygonlabels",
                "value": {"points": final_percent, "polygonlabels": ["car"], "closed": True},
            }
        ],
        effort=NormalizedEffortScore(
            delta_e_raw=12.5,
            delta_e_norm=0.5,
            dropped_as_bot=False,
            population_mean=10.0,
            population_stddev=5.0,
        ),
        model_version="selfcheck-0.0.0",
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        label="car",
    )


def _find_schema() -> Path:
    for candidate in SCHEMA_PATHS:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"tier3_schema.sql not found in any of {[str(p) for p in SCHEMA_PATHS]}")


async def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s %(message)s")

    settings = load_settings()
    log_provisional_hyperparameters(settings)
    log.info("dsn=%s", settings.pg_dsn.rsplit("@", 1)[-1])  # host/db only, no credentials

    import asyncpg

    failures: List[str] = []

    schema_sql = _find_schema().read_text(encoding="utf-8")
    conn = await asyncpg.connect(dsn=settings.pg_dsn)
    try:
        await conn.execute(schema_sql)
        log.info("PASS  schema applied (idempotent)")

        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'tier3' ORDER BY table_name"
        )
        found = {r["table_name"] for r in tables}
        expected = {
            "processed_annotations",
            "effort_population_stats",
            "replay_buffer",
            "ppo_training_runs",
        }
        missing = expected - found
        if missing:
            failures.append(f"missing tier3 tables: {sorted(missing)}")
        else:
            log.info("PASS  all four tier3 tables present")

        columns = {
            r["column_name"]
            for r in await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema='tier3' AND table_name='replay_buffer'"
            )
        }
        for required in ("label", "is_honeypot", "model_version", "delta_iou", "alpha", "beta"):
            if required not in columns:
                failures.append(f"replay_buffer.{required} is missing")
        log.info("PASS  replay_buffer has the reward columns")
    finally:
        await conn.close()

    # --- the reward path, against real Postgres --------------------------
    annotation_id = f"selfcheck-{uuid.uuid4().hex[:12]}"
    task_id = f"selfcheck-task-{uuid.uuid4().hex[:8]}"
    writer = PostgresReplayBuffer(settings)
    stage2.configure(settings=settings, writer=writer)

    try:
        await stage2.process_stage1(_probe_stage1(annotation_id, task_id))
        await stage2.process_stage1(_probe_stage1(annotation_id, task_id))  # replay

        conn = await asyncpg.connect(dsn=settings.pg_dsn)
        try:
            rows = await conn.fetch(
                "SELECT tuple_id, reward_r_t, delta_iou, model_version, label, is_honeypot "
                "FROM tier3.replay_buffer WHERE annotation_id = $1",
                annotation_id,
            )
            if len(rows) == 1:
                log.info(
                    "PASS  replayed annotation wrote exactly one row "
                    "(delta_iou=%.6f r_t=%.6f)",
                    rows[0]["delta_iou"], rows[0]["reward_r_t"],
                )
            else:
                failures.append(f"expected 1 row for a replayed annotation, found {len(rows)}")

            # Hand-checked against the unit suite: 1 - 0.21/0.25875 = 0.1884057971
            if rows and abs(rows[0]["delta_iou"] - 0.18840579710144931) > 1e-9:
                failures.append(
                    f"delta_iou {rows[0]['delta_iou']} does not match the hand-computed "
                    "0.1884057971 for the fixture geometry"
                )
            else:
                log.info("PASS  delta_iou matches the hand-computed fixture value")
        finally:
            await conn.close()
    finally:
        await writer.close()
        conn = await asyncpg.connect(dsn=settings.pg_dsn)
        try:
            deleted = await conn.execute(
                "DELETE FROM tier3.replay_buffer WHERE annotation_id = $1", annotation_id
            )
            log.info("cleanup  %s", deleted)
        finally:
            await conn.close()

    if failures:
        for failure in failures:
            log.error("FAIL  %s", failure)
        return 1

    log.info("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
