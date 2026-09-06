"""
Exact polygon geometry for the reward path.

Two jobs: get every mask into one coordinate frame, and compute an IoU that is
exact rather than approximate.

**Why not reuse `serving_ui/app/geometry.raster_iou`.** Its own docstring bars
it: "This is NOT the Tier 3 delta-IoU: that one is exact, lives in the E-DRDE
engine... Do not import this from a reward path." It rasterises onto a 256x256
grid, which is fine for asserting a wiggle moved the mask and not fine for a
number that gets multiplied by alpha and backpropagated. So this module uses
shapely/GEOS for true polygon boolean ops.

**The coordinate frame.** Three masks meet here and they do not arrive in the
same units:

  - `m_initial`, `m_wiggled`, `m_gold` — absolute pixels (Tier 1/2 convention).
  - `M_final`, parsed out of the Label Studio webhook — percentages of the
    image, 0-100.

Everything is converted to a normalised unit frame, x/W and y/H for pixels,
p/100 for percentages. IoU is invariant under that map — it is affine, so
intersection and union areas both scale by the same determinant and the ratio
survives — which means the unit-frame IoU equals the pixel-frame IoU exactly.
Normalising rather than converting to pixels also keeps the numbers small and
well-conditioned for GEOS.

**Invalid rings are repaired, not tolerated silently.** A Gaussian-perturbed
polygon can self-intersect, and hand-drawn ones routinely do. GEOS returns
garbage areas for invalid input, so every ring goes through `make_valid`. If a
ring cannot be repaired into positive area, that is an error the caller must
drop the tuple over — never a zero that quietly poisons the reward.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.validation import make_valid

log = logging.getLogger(__name__)

Point = Sequence[float]
Points = List[List[float]]

# Label Studio expresses polygon vertices as percentages of the image.
LS_PERCENT_SCALE = 100.0


class GeometryError(ValueError):
    """A mask could not be turned into positive-area geometry."""


def as_array(points: Sequence[Point]) -> np.ndarray:
    """(N, 2) float64. Raises on anything that cannot bound an area."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise GeometryError(f"expected an (N, 2) point list, got shape {arr.shape}")
    if arr.shape[0] < 3:
        raise GeometryError(f"a polygon needs at least 3 vertices, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise GeometryError("polygon contains NaN or infinite coordinates")
    return arr


def pixels_to_unit(points: Sequence[Point], width: float, height: float) -> np.ndarray:
    """Absolute pixels -> the unit frame. `width`/`height` are the image dimensions."""
    if width <= 0 or height <= 0:
        raise GeometryError(f"image dimensions must be positive, got {width}x{height}")
    return as_array(points) / np.array([width, height], dtype=np.float64)


def percent_to_unit(points: Sequence[Point]) -> np.ndarray:
    """Label Studio percentages (0-100) -> the unit frame."""
    return as_array(points) / LS_PERCENT_SCALE


def to_polygon(arr: np.ndarray) -> BaseGeometry:
    """
    Build a valid, positive-area polygon from a ring.

    Self-intersecting rings are repaired with `make_valid`, which can hand back
    a MultiPolygon or a GeometryCollection; only the polygonal parts carry area,
    so lines and points from the repair are discarded.

    The area check has to come *after* the repair, never before. A bowtie's raw
    shoelace area is 0.0 because its two lobes cancel, so an early area test
    would reject exactly the self-intersecting input this repair exists to
    rescue — `make_valid` turns that same bowtie into two triangles of positive
    area.
    """
    poly = Polygon(arr)
    if not poly.is_valid:
        repaired = _polygonal_parts(make_valid(poly))
        if repaired is None:
            # `make_valid` returned only lines or points: the ring bounds nothing.
            raise GeometryError("polygon encloses zero area (collinear or degenerate vertices)")
        poly = repaired
    if poly.is_empty or poly.area <= 0.0:
        raise GeometryError("polygon encloses zero area (collinear or degenerate vertices)")
    return poly


def _polygonal_parts(geom: BaseGeometry) -> BaseGeometry | None:
    """Reduce a repair result to its area-bearing parts, or None if it has none."""
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        return geom if not geom.is_empty else None
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))]
    if not parts:
        return None
    merged = unary_union(parts)
    return merged if not merged.is_empty else None


def exact_iou(a: BaseGeometry, b: BaseGeometry) -> float:
    """
    True Intersection-over-Union of two polygonal geometries.

        IoU(A, B) = |A ∩ B| / |A ∪ B|

    Disjoint masks give 0.0. The result is clamped into [0, 1] purely to absorb
    floating-point drift at the endpoints — `GeometricDelta` constrains both IoU
    fields to that interval and a 1.0000000002 would fail validation.
    """
    union = a.union(b).area
    if union <= 0.0:
        return 0.0
    return float(min(1.0, max(0.0, a.intersection(b).area / union)))


# ---------------------------------------------------------------------------
# Parsing M_final out of the Label Studio result array
# ---------------------------------------------------------------------------

# `type` values that carry a polygon ring in `value.points`.
_POLYGON_TYPES = ("polygonlabels", "polygon")


def _is_polygon_region(region: Dict[str, Any]) -> bool:
    if not isinstance(region, dict):
        return False
    if str(region.get("type", "")).lower() not in _POLYGON_TYPES:
        return False
    value = region.get("value")
    return isinstance(value, dict) and bool(value.get("points"))


def region_label(region: Dict[str, Any]) -> str | None:
    value = region.get("value") or {}
    labels = value.get("polygonlabels") or value.get("labels") or []
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return None


def extract_final_mask(
    ls_result: Iterable[Dict[str, Any]],
    label: str | None = None,
) -> Tuple[BaseGeometry, Points, str]:
    """
    Build `M_final` from the annotator's submitted regions, in the unit frame.

    Returns `(geometry, points_percent, strategy)` where `strategy` names how
    the regions were selected, so the caller can log it.

    Region selection, in order:

      1. If a `label` is supplied and some polygon regions carry it, use those.
         The served prediction is a single labelled region, so this keeps a
         second object the annotator drew from being merged into the reward.
      2. Exactly one polygon region — use it.
      3. Several — union them. An annotator splitting one object into two
         polygons is ordinary; the union is the mask they meant.

    Region identity would be the better discriminator, but it is not available:
    Tier 2 tags each served region with `meta.text = "wiggle_seed=..."`, and the
    gateway's re-parse through `LSResultRegion` (id/type/value only, extras
    ignored) drops `meta` before the envelope reaches Redis. Label is the
    strongest signal that survives.
    """
    regions = [r for r in ls_result if _is_polygon_region(r)]
    if not regions:
        raise GeometryError(
            "annotation result contains no polygon regions — nothing to compare M_final against"
        )

    strategy = "single_region"
    if label:
        matching = [r for r in regions if region_label(r) == label]
        if matching and len(matching) != len(regions):
            regions, strategy = matching, "label_match"
        elif matching:
            regions = matching

    if len(regions) > 1:
        strategy = "union_all" if strategy != "label_match" else "union_label_match"

    geoms = [to_polygon(percent_to_unit(r["value"]["points"])) for r in regions]
    merged = unary_union(geoms) if len(geoms) > 1 else geoms[0]

    reduced = _polygonal_parts(merged)
    if reduced is None or reduced.area <= 0.0:
        raise GeometryError("merged M_final regions enclose zero area")

    # Percent-space copy of the vertices, for logging and debugging only.
    points_percent: Points = [[float(x), float(y)] for r in regions for x, y in r["value"]["points"]]
    return reduced, points_percent, strategy
