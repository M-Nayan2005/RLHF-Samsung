"""
Polygon geometry helpers for the stochastic mask decoder.

Deliberately dependency-light: numpy only, no shapely. Everything here operates
on absolute-pixel polygons in the `PolygonMask.points` shape ([[x, y], ...]).

The canonicalisation in this module is load-bearing for reproducibility. See
`wiggle.py` and `services/serving_ui/README.md` -> "Reproducibility contract".
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

Point = Sequence[float]
Points = List[List[float]]


# --------------------------------------------------------------------------
# Basic measures
# --------------------------------------------------------------------------

def as_array(points: Sequence[Point]) -> np.ndarray:
    """(N, 2) float64 array. Raises on a degenerate polygon."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) point list, got shape {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError(f"a polygon needs at least 3 points, got {arr.shape[0]}")
    return arr


def as_points(arr: np.ndarray, ndigits: int = 4) -> Points:
    """Back to the JSON-friendly nested-list shape, rounded for stable output."""
    return [[round(float(x), ndigits), round(float(y), ndigits)] for x, y in arr]


def signed_area(arr: np.ndarray) -> float:
    """Shoelace. Positive = counter-clockwise in a y-down image coordinate frame."""
    x, y = arr[:, 0], arr[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def area(arr: np.ndarray) -> float:
    return abs(signed_area(arr))


def centroid(arr: np.ndarray) -> np.ndarray:
    """
    Area-weighted polygon centroid, not the vertex mean. The vertex mean drifts
    toward whichever edge happens to carry more vertices, which would make the
    wiggle's rotation and scale pivot depend on vertex density.

    Falls back to the vertex mean for a zero-area (collinear) polygon.
    """
    a = signed_area(arr)
    if abs(a) < 1e-12:
        return arr.mean(axis=0)
    x, y = arr[:, 0], arr[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    cx = float(np.dot(x + xn, cross)) / (6.0 * a)
    cy = float(np.dot(y + yn, cross)) / (6.0 * a)
    return np.array([cx, cy], dtype=np.float64)


def bbox(arr: np.ndarray) -> Tuple[float, float, float, float]:
    return (
        float(arr[:, 0].min()),
        float(arr[:, 1].min()),
        float(arr[:, 0].max()),
        float(arr[:, 1].max()),
    )


def diagonal(x_min: float, y_min: float, x_max: float, y_max: float) -> float:
    """Length of the bounding-box diagonal. The wiggle's length scale."""
    return float(np.hypot(x_max - x_min, y_max - y_min))


# --------------------------------------------------------------------------
# Canonicalisation - required for seed reproducibility
# --------------------------------------------------------------------------

def canonicalize(arr: np.ndarray) -> np.ndarray:
    """
    Put a polygon into a canonical form so the same geometry always consumes the
    same random draws in the same order.

    Three normalisations, in order:

    1. Drop a duplicated closing vertex (some producers repeat p[0] at the end).
    2. Force counter-clockwise winding.
    3. Rotate the vertex list to start at the lexicographically smallest (x, y).

    Without this, an upstream service that reverses its winding or starts the
    ring at a different vertex would produce a different wiggled mask from the
    same `wiggle_seed`, and Tier 3 could no longer reconstruct the action A_t.
    """
    if arr.shape[0] >= 4 and np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    if arr.shape[0] < 3:
        raise ValueError("polygon collapsed to fewer than 3 points after closing-vertex removal")

    if signed_area(arr) < 0:
        arr = arr[::-1]

    order = np.lexsort((arr[:, 1], arr[:, 0]))
    start = int(order[0])
    if start:
        arr = np.roll(arr, -start, axis=0)

    return np.ascontiguousarray(arr, dtype=np.float64)


# --------------------------------------------------------------------------
# Outward normals - used by the boundary-offset action dimension
# --------------------------------------------------------------------------

def _normalize_rows(v: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(v, axis=1, keepdims=True)
    return np.divide(v, np.maximum(lengths, 1e-12))


def outward_normals(arr: np.ndarray) -> np.ndarray:
    """
    Unit outward normal per vertex, for a counter-clockwise ring in a y-down
    image frame. Averages the two adjacent edge normals so the result is stable
    at corners.

    Degenerate vertices (a zero-length averaged normal, e.g. a 180-degree spike)
    fall back to the radial direction from the centroid.
    """
    prev_pt = np.roll(arr, 1, axis=0)
    next_pt = np.roll(arr, -1, axis=0)

    e_in = arr - prev_pt
    e_out = next_pt - arr

    # Right-hand perpendicular of a CCW edge points outward in a y-down frame.
    n_in = np.stack([e_in[:, 1], -e_in[:, 0]], axis=1)
    n_out = np.stack([e_out[:, 1], -e_out[:, 0]], axis=1)

    n = _normalize_rows(n_in) + _normalize_rows(n_out)
    lengths = np.linalg.norm(n, axis=1, keepdims=True)

    degenerate = lengths[:, 0] < 1e-9
    if degenerate.any():
        radial = arr[degenerate] - centroid(arr)
        n[degenerate] = _normalize_rows(radial)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)

    return np.divide(n, np.maximum(lengths, 1e-12))


# --------------------------------------------------------------------------
# Clamping
# --------------------------------------------------------------------------

def clamp_to_frame(arr: np.ndarray, width: float, height: float) -> np.ndarray:
    """
    Keep vertices inside the image. Label Studio silently discards a region whose
    percentage coordinates fall outside [0, 100], so an unclamped wiggle near an
    image edge would drop the pre-annotation entirely, and the annotator would
    draw from scratch: telemetry for a task that was never really served.
    """
    out = arr.copy()
    np.clip(out[:, 0], 0.0, float(width), out=out[:, 0])
    np.clip(out[:, 1], 0.0, float(height), out=out[:, 1])
    return out


# --------------------------------------------------------------------------
# IoU - local sanity checks only
# --------------------------------------------------------------------------

def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorised even-odd ray casting: (M,2) points against one (N,2) ring."""
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(pts.shape[0], dtype=bool)

    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)

    for i in range(poly.shape[0]):
        straddles = (y1[i] > y) != (y2[i] > y)
        denom = y2[i] - y1[i]
        if abs(denom) < 1e-12:
            continue
        x_cross = (x2[i] - x1[i]) * (y - y1[i]) / denom + x1[i]
        inside ^= straddles & (x < x_cross)

    return inside


def raster_iou(a: np.ndarray, b: np.ndarray, resolution: int = 256) -> float:
    """
    Approximate IoU by rasterising both polygons onto a shared grid.

    Used only by this service's tests and the /wiggle/preview debug endpoint, to
    assert the wiggle actually moved the mask by a sane amount. This is NOT the
    Tier 3 delta-IoU: that one is exact, lives in the E-DRDE engine, and is out
    of scope for the Tier 1-2 MVP. Do not import this from a reward path.
    """
    x_min = min(a[:, 0].min(), b[:, 0].min())
    y_min = min(a[:, 1].min(), b[:, 1].min())
    x_max = max(a[:, 0].max(), b[:, 0].max())
    y_max = max(a[:, 1].max(), b[:, 1].max())
    if x_max - x_min < 1e-9 or y_max - y_min < 1e-9:
        return 0.0

    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    in_a = _points_in_polygon(pts, a)
    in_b = _points_in_polygon(pts, b)

    union = int(np.count_nonzero(in_a | in_b))
    if union == 0:
        return 0.0
    return int(np.count_nonzero(in_a & in_b)) / union
