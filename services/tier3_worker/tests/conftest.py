"""
Shared fixtures for the Dev 2 suite.

Fixes `sys.path` so the suite runs from the repo root without installation.
Inside the container `/app` is the working directory and both `tier3_worker`
and `common` resolve; on a laptop the layout is `services/tier3_worker` +
`common/`, so both roots go on the path here. Same approach as
`services/serving_ui/tests/conftest.py`.

Every test in this suite is offline — no Postgres, no Redis, no network.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_ROOT = REPO_ROOT / "services"

for entry in (REPO_ROOT, SERVICES_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from common.schemas.tier1_ingestion import PolygonMask  # noqa: E402
from common.schemas.tier3_rlhf import NormalizedEffortScore, Stage1Output  # noqa: E402
from tier3_worker.config import Settings  # noqa: E402

# One fixture image, used everywhere so the arithmetic in the tests is checkable
# by hand. Deliberately non-square: a square image would hide a swapped width
# and height, which is exactly the bug the unit-frame conversion exists to avoid.
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 400


def rect(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
    """Axis-aligned rectangle, counter-clockwise in a y-down frame."""
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def rect_percent(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
    """The same rectangle given in pixels, expressed as Label Studio percentages."""
    return [
        [x / IMAGE_WIDTH * 100.0, y / IMAGE_HEIGHT * 100.0]
        for x, y in rect(x0, y0, x1, y1)
    ]


# M_initial: the Tier 1 consensus mask. M_wiggled: the sampled action, offset
# from it. Both in absolute pixels, as Tier 1/2 store them.
BASELINE = rect(100, 50, 400, 300)
WIGGLED = rect(120, 60, 420, 310)


def ls_region(
    points_percent: List[List[float]],
    label: str = "car",
    region_id: str = "region_1",
    region_type: str = "polygonlabels",
) -> Dict[str, Any]:
    """
    One entry of `payload.result`, shaped the way it actually arrives.

    Note what is absent: `original_width`, `original_height` and `meta`. Tier 2
    sets all three on the served region, but the gateway re-parses the body
    through `LSResultRegion` (id/type/value only, pydantic v2 ignores extras),
    so they are gone before Tier 3 sees the envelope. The fixture reflects the
    real input, not the ideal one.
    """
    return {
        "id": region_id,
        "type": region_type,
        "value": {"points": points_percent, "polygonlabels": [label], "closed": True},
    }


@pytest.fixture
def settings() -> Settings:
    """Defaults, with the provisional alpha/beta the Tier 3/4 plan nominates."""
    return Settings(
        alpha=1.0,
        beta=0.3,
        reference_mask="wiggled",
        use_gold_when_available=True,
        pg_dsn="postgresql://unused",
        db_statement_timeout_ms=5000,
        apply_schema_on_start=False,
        require_model_version=True,
    )


@pytest.fixture
def make_stage1():
    """
    Factory for `Stage1Output`, the object Dev 1 hands over.

    Defaults describe an ordinary annotation: the human corrected the wiggled
    mask back onto the baseline, at moderate effort.
    """

    def _make(
        *,
        ls_result: Optional[List[Dict[str, Any]]] = None,
        m_initial: Optional[List[List[float]]] = None,
        m_wiggled: Optional[List[List[float]]] = None,
        delta_e_norm: float = 0.5,
        dropped_as_bot: bool = False,
        model_version: Optional[str] = "serving-ui-stochastic-0.1.0",
        annotation_id: str = "ann_9f8e7d",
        task_id: str = "task_a1b2c3",
        is_honeypot: bool = False,
        m_gold: Optional[List[List[float]]] = None,
        label: Optional[str] = "car",
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ) -> Stage1Output:
        effort = NormalizedEffortScore(
            delta_e_raw=12.5,
            delta_e_norm=delta_e_norm,
            dropped_as_bot=dropped_as_bot,
            population_mean=10.0,
            population_stddev=5.0,
        )
        return Stage1Output(
            annotation_id=annotation_id,
            task_id=task_id,
            wiggle_seed="c644b8cf379b545910d076f8e05d913c",
            m_initial=PolygonMask(points=m_initial if m_initial is not None else BASELINE),
            m_wiggled=PolygonMask(points=m_wiggled if m_wiggled is not None else WIGGLED),
            ls_result=ls_result if ls_result is not None else [ls_region(rect_percent(*_bounds(BASELINE)))],
            effort=effort,
            dropped_as_bot=dropped_as_bot,
            model_version=model_version,
            image_width=image_width,
            image_height=image_height,
            is_honeypot=is_honeypot,
            m_gold=PolygonMask(points=m_gold) if m_gold is not None else None,
            label=label,
        )

    return _make


def _bounds(points: List[List[float]]) -> tuple:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
