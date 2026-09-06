"""
Geometry and coordinate-frame tests.

The load-bearing one is `test_iou_is_invariant_under_the_percent_conversion`.
The whole design rests on being able to compare a pixel-space mask against a
percent-space one by normalising both, and that is only legitimate because IoU
survives the anisotropic scale between the two frames.
"""
from __future__ import annotations

import math

import pytest

from tier3_worker.geometry import (
    GeometryError,
    as_array,
    exact_iou,
    extract_final_mask,
    percent_to_unit,
    pixels_to_unit,
    region_label,
    to_polygon,
)

from .conftest import IMAGE_HEIGHT, IMAGE_WIDTH, ls_region, rect, rect_percent


def poly_px(x0, y0, x1, y1):
    return to_polygon(as_array(rect(x0, y0, x1, y1)))


def poly_unit(x0, y0, x1, y1):
    return to_polygon(pixels_to_unit(rect(x0, y0, x1, y1), IMAGE_WIDTH, IMAGE_HEIGHT))


# ---------------------------------------------------------------------------
# exact_iou
# ---------------------------------------------------------------------------

def test_identical_masks_score_one():
    a = poly_px(0, 0, 10, 10)
    assert exact_iou(a, a) == pytest.approx(1.0)


def test_disjoint_masks_score_zero():
    assert exact_iou(poly_px(0, 0, 10, 10), poly_px(100, 100, 110, 110)) == 0.0


def test_half_overlap_is_exact_not_approximate():
    """
    Two unit squares offset by half their width: intersection 0.5, union 1.5,
    IoU exactly 1/3. A rasterised IoU lands near this; an exact one hits it.
    """
    iou = exact_iou(poly_px(0, 0, 1, 1), poly_px(0.5, 0, 1.5, 1))
    assert iou == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_containment():
    """A mask fully inside another scores area_inner / area_outer."""
    iou = exact_iou(poly_px(0, 0, 10, 10), poly_px(2, 2, 4, 4))
    assert iou == pytest.approx(4.0 / 100.0, abs=1e-12)


def test_iou_is_invariant_under_the_percent_conversion():
    """
    The claim the whole coordinate strategy depends on.

    Pixels -> percent is diag(100/W, 100/H): affine and invertible, so
    intersection and union areas both scale by the same determinant and their
    ratio is unchanged. If this ever fails, comparing M_final against a cached
    pixel mask is invalid and every reward in the buffer is wrong.
    """
    a_px, b_px = rect(100, 50, 400, 300), rect(120, 60, 420, 310)

    in_pixels = exact_iou(to_polygon(as_array(a_px)), to_polygon(as_array(b_px)))
    in_unit = exact_iou(
        to_polygon(pixels_to_unit(a_px, IMAGE_WIDTH, IMAGE_HEIGHT)),
        to_polygon(pixels_to_unit(b_px, IMAGE_WIDTH, IMAGE_HEIGHT)),
    )
    in_percent = exact_iou(
        to_polygon(percent_to_unit(rect_percent(*_b(a_px)))),
        to_polygon(percent_to_unit(rect_percent(*_b(b_px)))),
    )

    assert in_pixels == pytest.approx(in_unit, abs=1e-12)
    assert in_pixels == pytest.approx(in_percent, abs=1e-12)


def test_swapped_dimensions_change_the_shape():
    """
    A non-square image is used throughout precisely so that transposing width
    and height is detectable.

    Note what is *not* asserted: area. Normalising by (1/W, 1/H) and by
    (1/H, 1/W) scales area by the same 1/(W·H) either way, so the two areas are
    identical and an area comparison would pass while the mask sat in entirely
    the wrong place. The shape is what moves, so the shape is what is checked.
    """
    pts = rect(100, 50, 400, 300)
    right = to_polygon(pixels_to_unit(pts, IMAGE_WIDTH, IMAGE_HEIGHT))
    wrong = to_polygon(pixels_to_unit(pts, IMAGE_HEIGHT, IMAGE_WIDTH))

    assert right.area == pytest.approx(wrong.area, abs=1e-12)  # the trap
    assert exact_iou(right, wrong) < 0.5                        # the real check
    assert not math.isclose(right.bounds[2] - right.bounds[0],
                            wrong.bounds[2] - wrong.bounds[0])


# ---------------------------------------------------------------------------
# validity and degenerate input
# ---------------------------------------------------------------------------

def test_two_point_polygon_is_rejected():
    with pytest.raises(GeometryError, match="at least 3 vertices"):
        as_array([[0.0, 0.0], [1.0, 1.0]])


def test_collinear_polygon_encloses_no_area():
    with pytest.raises(GeometryError, match="zero area"):
        to_polygon(as_array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))


def test_non_finite_coordinates_are_rejected():
    with pytest.raises(GeometryError, match="NaN or infinite"):
        as_array([[0.0, 0.0], [float("nan"), 1.0], [2.0, 2.0]])


def test_self_intersecting_bowtie_is_repaired():
    """
    A wiggled polygon can cross itself, and so can a hand-drawn one. GEOS
    returns a meaningless area for an invalid ring, so the repair has to happen
    before any IoU is taken.
    """
    bowtie = as_array([[0.0, 0.0], [10.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    repaired = to_polygon(bowtie)
    assert repaired.is_valid
    # Two triangles of area 25 each; the crossing does not double-count.
    assert repaired.area == pytest.approx(50.0, abs=1e-9)


def test_zero_dimensions_are_rejected():
    with pytest.raises(GeometryError, match="dimensions must be positive"):
        pixels_to_unit(rect(0, 0, 1, 1), 0, 100)


# ---------------------------------------------------------------------------
# extract_final_mask
# ---------------------------------------------------------------------------

def test_single_region_is_used_directly():
    result = [ls_region(rect_percent(100, 50, 400, 300))]
    geom, _points, strategy = extract_final_mask(result, "car")
    assert strategy == "single_region"
    assert geom.area == pytest.approx(poly_unit(100, 50, 400, 300).area, abs=1e-12)


def test_no_polygon_regions_raises():
    with pytest.raises(GeometryError, match="no polygon regions"):
        extract_final_mask([{"id": "r1", "type": "choices", "value": {"choices": ["yes"]}}], None)


def test_empty_result_raises():
    with pytest.raises(GeometryError, match="no polygon regions"):
        extract_final_mask([], None)


def test_split_object_is_unioned():
    """An annotator splitting one object into two polygons meant their union."""
    result = [
        ls_region(rect_percent(100, 50, 200, 300), region_id="r1"),
        ls_region(rect_percent(300, 50, 400, 300), region_id="r2"),
    ]
    geom, _points, strategy = extract_final_mask(result, "car")
    assert strategy == "union_all"
    expected = poly_unit(100, 50, 200, 300).area + poly_unit(300, 50, 400, 300).area
    assert geom.area == pytest.approx(expected, abs=1e-12)


def test_a_second_object_with_another_label_is_excluded():
    """
    The reward must describe the region the policy served. A truck the
    annotator also drew is not part of the car's correction cost.
    """
    result = [
        ls_region(rect_percent(100, 50, 400, 300), label="car", region_id="r1"),
        ls_region(rect_percent(600, 50, 700, 300), label="truck", region_id="r2"),
    ]
    geom, _points, strategy = extract_final_mask(result, "car")
    assert strategy == "label_match"
    assert geom.area == pytest.approx(poly_unit(100, 50, 400, 300).area, abs=1e-12)


def test_unknown_label_falls_back_to_all_regions():
    """
    A label that matches nothing must not empty the mask. Better to union what
    is there and log it than to drop a real annotation over a label mismatch.
    """
    result = [ls_region(rect_percent(100, 50, 400, 300), label="car")]
    geom, _points, strategy = extract_final_mask(result, "bicycle")
    assert strategy == "single_region"
    assert geom.area > 0


def test_plain_polygon_type_is_accepted():
    result = [ls_region(rect_percent(100, 50, 400, 300), region_type="polygon")]
    geom, _points, _strategy = extract_final_mask(result, None)
    assert geom.area > 0


def test_region_label_reads_polygonlabels():
    assert region_label(ls_region(rect_percent(0, 0, 10, 10), label="bus")) == "bus"
    assert region_label({"value": {}}) is None


def _b(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))
