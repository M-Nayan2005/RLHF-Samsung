"""
Tests for the Label Studio formatting layer.

Most of what is checked here fails *silently* in production if it regresses:
Label Studio discards a malformed or mismatched region without surfacing an
error, so the annotator simply sees an empty canvas and assumes the model had
nothing to say. These are the assertions that turn that into a red test.
"""
from __future__ import annotations

import pytest

from app import ls_format
from app.config import load_settings
from app.image_meta import ImageDims
from app.wiggle import apply_wiggle

XML = """
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="polygon" toName="image">
    <Label value="car"/>
    <Label value="object"/>
  </PolygonLabels>
</View>
"""


@pytest.fixture
def settings(base_env):
    return load_settings()


@pytest.fixture
def dims():
    return ImageDims(800, 600, "fixed")


@pytest.fixture
def wiggled(routing_task):
    box = routing_task.bounding_box
    return apply_wiggle(
        routing_task.baseline_mask.points,
        seed="fixed-seed",
        sigma=0.02,
        bounding_box=[box.x_min, box.y_min, box.x_max, box.y_max],
        image_width=800,
        image_height=600,
    )


# ----------------------------------------------------------------------
# label_config parsing
# ----------------------------------------------------------------------

def test_tag_names_come_from_the_live_config(settings):
    info = ls_format.parse_label_config(XML, settings)
    assert (info.from_name, info.to_name) == ("polygon", "image")
    assert info.labels == ["car", "object"]


def test_the_shipped_labeling_config_parses(settings, label_config):
    """The XML in the repo must be readable by the code that consumes it."""
    info = ls_format.parse_label_config(label_config, settings)
    assert info.from_name == settings.ls_from_name
    assert info.to_name == settings.ls_to_name
    assert settings.ls_label_fallback in info.labels


def test_missing_config_falls_back_to_env(settings):
    info = ls_format.parse_label_config(None, settings)
    assert info.from_name == settings.ls_from_name


def test_malformed_config_falls_back_rather_than_raising(settings):
    """A broken config must not take /predict down; a blank canvas is worse than a stale tag name."""
    info = ls_format.parse_label_config("<View><not closed", settings)
    assert info.from_name == settings.ls_from_name


def test_config_without_a_polygon_control_falls_back(settings):
    info = ls_format.parse_label_config(
        '<View><Image name="i" value="$image"/><Choices name="c" toName="i"/></View>', settings
    )
    assert info.from_name == settings.ls_from_name


# ----------------------------------------------------------------------
# Coordinate conversion
# ----------------------------------------------------------------------

def test_pixels_convert_to_percentages():
    assert ls_format.to_percent([[400, 300]], 800, 600) == [[50.0, 50.0]]


def test_percent_conversion_round_trips():
    original = [[123.5, 456.25], [700.0, 12.0]]
    back = ls_format.from_percent(ls_format.to_percent(original, 800, 600), 800, 600)
    for (ax, ay), (bx, by) in zip(original, back):
        assert ax == pytest.approx(bx, abs=0.05)
        assert ay == pytest.approx(by, abs=0.05)


def test_out_of_range_points_are_clipped_not_dropped():
    """Label Studio discards a region with any coordinate outside [0, 100]."""
    points = ls_format.to_percent([[-50, -50], [1600, 1200]], 800, 600)
    assert points == [[0.0, 0.0], [100.0, 100.0]]


def test_zero_dimensions_are_rejected():
    with pytest.raises(ValueError):
        ls_format.to_percent([[1, 1]], 0, 600)


# ----------------------------------------------------------------------
# Prediction assembly
# ----------------------------------------------------------------------

def test_region_carries_the_tags_label_studio_matches_on(routing_task, wiggled, dims, settings):
    info = ls_format.parse_label_config(XML, settings)
    bundle = ls_format.build_prediction(routing_task, wiggled, dims, settings, info)
    region = bundle.prediction.result[0]

    assert region["from_name"] == "polygon"
    assert region["to_name"] == "image"
    assert region["type"] == "polygonlabels"
    assert region["original_width"] == 800
    assert region["original_height"] == 600
    assert region["value"]["closed"] is True
    assert len(region["value"]["points"]) == len(wiggled.points)


def test_grounding_dino_label_is_preserved_when_declared(routing_task, wiggled, dims, settings):
    info = ls_format.parse_label_config(XML, settings)
    bundle = ls_format.build_prediction(routing_task, wiggled, dims, settings, info)
    assert bundle.label == "car"


def test_undeclared_label_falls_back_instead_of_vanishing(routing_task, wiggled, dims, settings):
    """A region with an undeclared label is dropped by Label Studio with no error."""
    task = routing_task.model_copy(
        update={"bounding_box": routing_task.bounding_box.model_copy(update={"label": "spaceship"})}
    )
    info = ls_format.parse_label_config(XML, settings)
    bundle = ls_format.build_prediction(task, wiggled, dims, settings, info)
    assert bundle.label == settings.ls_label_fallback


def test_empty_prediction_has_no_regions(settings):
    assert ls_format.empty_prediction(settings).result == []


# ----------------------------------------------------------------------
# Seed round-trip — all three channels
# ----------------------------------------------------------------------

def test_seed_survives_in_model_version(routing_task, wiggled, dims, settings):
    info = ls_format.parse_label_config(XML, settings)
    bundle = ls_format.build_prediction(routing_task, wiggled, dims, settings, info)
    assert ls_format.decode_model_version(bundle.prediction.model_version) == wiggled.params.seed


def test_seed_survives_in_region_meta(routing_task, wiggled, dims, settings):
    info = ls_format.parse_label_config(XML, settings)
    bundle = ls_format.build_prediction(routing_task, wiggled, dims, settings, info)
    assert ls_format.seed_from_region_meta(bundle.prediction.result[0]) == wiggled.params.seed


def test_model_version_without_a_seed_decodes_to_none():
    assert ls_format.decode_model_version("serving-ui-stochastic-0.1.0") is None


def test_region_without_meta_decodes_to_none():
    assert ls_format.seed_from_region_meta({"value": {}}) is None


def test_task_data_carries_the_queue_task_id(routing_task, dims):
    data = ls_format.task_data_for_ls(routing_task, dims)
    assert data["task_id"] == routing_task.task_id
    assert data["image"] == routing_task.image_url
    assert data["width"] == 800
