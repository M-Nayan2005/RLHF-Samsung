"""
Tests for the stochastic policy.

The reproducibility properties get the most attention here, because they are the
ones nothing else in the system would notice breaking. A wiggle that stops being
reproducible from its seed still serves a perfectly good-looking mask, still
collects telemetry, and still submits — it just quietly severs the link between
the reward and the action that earned it, months before Tier 3 exists to notice.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app import geometry
from app.wiggle import (
    WIGGLE_ALGORITHM_VERSION,
    WiggleError,
    WiggleParams,
    apply_wiggle,
    new_seed,
    replay,
    rng_from_seed,
    seed_to_int,
)

SQUARE = [[100.0, 50.0], [400.0, 50.0], [400.0, 300.0], [100.0, 300.0]]
BOX = [100.0, 50.0, 400.0, 300.0]


def wiggle(points=None, **kwargs):
    params = {
        "seed": "fixed-seed-for-tests",
        "sigma": 0.02,
        "bounding_box": BOX,
        "image_width": 800,
        "image_height": 600,
    }
    params.update(kwargs)
    return apply_wiggle(points or SQUARE, **params)


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------

def test_same_seed_gives_identical_output():
    assert wiggle().points == wiggle().points


def test_different_seeds_give_different_output():
    assert wiggle(seed="seed-a").points != wiggle(seed="seed-b").points


def test_seed_derivation_is_stable_across_processes():
    """
    Pinned by value, not just by self-consistency.

    `hash()` would pass an equality check within one process and produce a
    different stream in the next, silently invalidating every seed already
    issued. This asserts the actual BLAKE2b-derived integer.
    """
    assert seed_to_int("seed_7781") == 0xE33550D55344B94A0FDD7EC49C5B3E42
    assert seed_to_int("canary") == 0x14D72801268839747D98EADB455479AD
    assert seed_to_int("abc") != seed_to_int("abd")
    assert 0 <= seed_to_int("abc") < 2 ** 128


def test_rng_stream_is_pinned_to_pcg64():
    """If this fails, numpy changed its Generator stream and every issued seed is void."""
    draws = rng_from_seed("canary").standard_normal(5)
    expected = np.random.Generator(np.random.PCG64(seed_to_int("canary"))).standard_normal(5)
    assert np.array_equal(draws, expected)


def test_non_hex_seeds_are_accepted():
    """The shared fixtures use `seed_7781`, which `int(seed, 16)` would reject."""
    assert wiggle(seed="seed_7781").points


def test_replay_reproduces_the_served_polygon():
    result = wiggle()
    again = replay(SQUARE, result.params, bounding_box=BOX, image_width=800, image_height=600)
    assert again.points == result.points


def test_replay_refuses_a_foreign_algorithm_version():
    params = WiggleParams(
        seed="s", sigma=0.02, mode="affine", scale_reference="bbox_diagonal",
        length_scale=390.5, clamped=False, algorithm_version="wiggle-0.0.1-ancient",
    )
    with pytest.raises(WiggleError, match="cannot replay"):
        replay(SQUARE, params, bounding_box=BOX)


def test_algorithm_version_is_recorded_on_every_result():
    assert wiggle().params.algorithm_version == WIGGLE_ALGORITHM_VERSION


# ----------------------------------------------------------------------
# Canonicalisation — the invariants reproducibility depends on
# ----------------------------------------------------------------------

def test_reversed_winding_produces_the_same_wiggle():
    assert wiggle(list(reversed(SQUARE))).points == wiggle().points


def test_rotated_vertex_order_produces_the_same_wiggle():
    assert wiggle(SQUARE[2:] + SQUARE[:2]).points == wiggle().points


def test_duplicated_closing_vertex_is_dropped():
    closed = SQUARE + [SQUARE[0]]
    result = wiggle(closed)
    assert len(result.points) == len(SQUARE)
    assert result.points == wiggle().points


# ----------------------------------------------------------------------
# Policy behaviour
# ----------------------------------------------------------------------

def test_sigma_zero_is_the_identity():
    """
    A deterministic decoder is Fault 3, so this is a debug setting, not a mode.
    It is tested because it is the only sigma with an exactly checkable answer.
    """
    result = wiggle(sigma=0.0)
    assert result.points == result.baseline_points


def test_larger_sigma_moves_the_mask_further():
    small = wiggle(sigma=0.01).diagnostics["mean_displacement_px"]
    large = wiggle(sigma=0.10).diagnostics["mean_displacement_px"]
    assert large > small


def test_default_sigma_lands_in_a_usable_range():
    """
    The wiggle has to be visible enough to correct and small enough to still be
    a near-miss. An IoU of 1.0 means nothing to fix (no exploration signal); a
    very low IoU means the annotator redraws from scratch and the correction no
    longer relates to the sampled action.
    """
    iou = wiggle(sigma=0.02).diagnostics["iou_vs_baseline"]
    assert 0.5 < iou < 0.99


def test_affine_action_is_five_dimensional():
    """Fault 2: the critic must attribute reward to a handful of parameters."""
    assert set(wiggle().action) == {
        "tx_px", "ty_px", "scale", "theta_rad", "normal_offset_px"
    }


def test_vertex_mode_perturbs_every_vertex():
    result = wiggle(mode="vertex")
    assert result.action["n_vertices"] == len(SQUARE)
    assert result.action["mean_vertex_shift_px"] > 0
    assert result.points != wiggle(mode="affine").points


def test_scale_stays_positive_under_extreme_sigma():
    """Scale is exponentiated, so the polygon can never invert through zero."""
    for seed in (f"s{i}" for i in range(40)):
        assert wiggle(seed=seed, sigma=2.0).action["scale"] > 0


def test_clamping_keeps_vertices_inside_the_frame():
    """Label Studio silently discards an out-of-range region, so this is load-bearing."""
    result = wiggle(sigma=0.9, seed="pushes-out", image_width=420, image_height=320)
    for x, y in result.points:
        assert 0.0 <= x <= 420.0
        assert 0.0 <= y <= 320.0
    assert result.params.clamped is True


def test_length_scale_comes_from_the_bounding_box():
    expected = math.hypot(400 - 100, 300 - 50)
    assert wiggle().params.length_scale == pytest.approx(expected)


def test_mask_diagonal_reference_ignores_the_box():
    result = wiggle(scale_reference="mask_diagonal", bounding_box=[0, 0, 10000, 10000])
    assert result.params.length_scale == pytest.approx(math.hypot(300, 250))


# ----------------------------------------------------------------------
# Degenerate input
# ----------------------------------------------------------------------

def test_too_few_points_is_rejected():
    with pytest.raises(WiggleError):
        wiggle([[0.0, 0.0], [1.0, 1.0]])


def test_negative_sigma_is_rejected():
    with pytest.raises(WiggleError):
        wiggle(sigma=-0.1)


def test_zero_extent_polygon_is_rejected():
    with pytest.raises(WiggleError, match="zero extent"):
        wiggle([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], bounding_box=None)


def test_zero_extent_polygon_is_rejected_even_with_a_healthy_bounding_box():
    """
    The length scale comes from Grounding DINO's box, so a collapsed mask paired
    with a healthy box would otherwise be wiggled into a zero-area polygon and
    served as an invisible region.
    """
    with pytest.raises(WiggleError, match="zero extent"):
        wiggle([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])


def test_collinear_polygon_is_rejected():
    with pytest.raises(WiggleError, match="zero area"):
        wiggle([[0.0, 0.0], [100.0, 100.0], [200.0, 200.0]])


def test_new_seed_is_unique_and_long_enough():
    seeds = {new_seed() for _ in range(200)}
    assert len(seeds) == 200
    assert all(len(s) == 32 for s in seeds)


def test_new_seed_rejects_a_short_length():
    with pytest.raises(WiggleError):
        new_seed(4)


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------

def test_area_weighted_centroid_ignores_vertex_density():
    """
    A vertex mean would drift toward the densely-sampled edge, which would make
    the rotation and scale pivot depend on how finely upstream sampled the mask.
    """
    dense = [[100.0, 50.0], [200.0, 50.0], [300.0, 50.0], [400.0, 50.0],
             [400.0, 300.0], [100.0, 300.0]]
    centroid = geometry.centroid(geometry.canonicalize(geometry.as_array(dense)))
    naive = np.asarray(dense).mean(axis=0)
    assert abs(centroid[1] - 175.0) < abs(naive[1] - 175.0)


def test_outward_normals_point_away_from_the_centre():
    arr = geometry.canonicalize(geometry.as_array(SQUARE))
    normals = geometry.outward_normals(arr)
    centre = geometry.centroid(arr)
    for point, normal in zip(arr, normals):
        radial = point - centre
        assert float(np.dot(radial / np.linalg.norm(radial), normal)) > 0


def test_raster_iou_is_one_for_identical_polygons():
    arr = geometry.as_array(SQUARE)
    assert geometry.raster_iou(arr, arr) == pytest.approx(1.0, abs=1e-9)


def test_raster_iou_is_zero_for_disjoint_polygons():
    a = geometry.as_array(SQUARE)
    b = geometry.as_array([[1000.0, 1000.0], [1100.0, 1000.0], [1100.0, 1100.0]])
    assert geometry.raster_iou(a, b) == pytest.approx(0.0, abs=1e-6)
