"""
End-to-end tests through the HTTP surface, in MOCK_MODE.

Two behaviours here matter more than the happy path.

**/predict must never return a non-200.** Label Studio treats an error from an
ML backend as a broken backend, shows the annotator a red banner, and stops
calling it for the rest of the session. One unroutable task must not end the
labeling session, so failures degrade to an empty prediction.

**The honeypot answer key must never reach the browser.** The plan makes this an
explicit merge criterion: `ground_truth_mask` must not appear in any response.
Dev 2 strips it; this asserts that serving_ui strips it again if they do not.
"""
from __future__ import annotations

import json

import pytest

from app.task_source import NoTaskAvailable, TaskSourceError

XML = """
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="polygon" toName="image">
    <Label value="car"/><Label value="object"/>
  </PolygonLabels>
</View>
"""


def predict(client, **overrides):
    body = {
        "tasks": [{"id": 7, "data": {"image": "http://example.invalid/i.jpg", "task_id": "task_a1b2c3"}}],
        "project": "1.16",
        "label_config": XML,
        "params": {"context": {"user": "annotator_42"}},
    }
    body.update(overrides)
    return client.post("/predict", json=body)


# ----------------------------------------------------------------------
# Protocol
# ----------------------------------------------------------------------

def test_health_reports_up_without_touching_upstream(client):
    """Label Studio polls this; it must not go red because routing_qa is down."""
    body = client.get("/health").json()
    assert body["status"] == "UP"
    assert "model_version" in body


def test_health_does_not_leak_the_api_token(client):
    assert client.get("/health").json()["config"]["label_studio_api_token"] in ("<set>", "<unset>")


def test_setup_returns_a_model_version(client):
    response = client.post("/setup", json={"project": "1.16", "label_config": XML})
    assert response.status_code == 200
    assert response.json()["model_version"]


def test_predict_returns_one_result_per_task(client):
    """Label Studio matches results to tasks positionally, so the count must match."""
    response = predict(client, tasks=[
        {"id": 1, "data": {"task_id": "a"}},
        {"id": 2, "data": {"task_id": "b"}},
    ])
    assert response.status_code == 200
    assert len(response.json()["results"]) == 2


def test_predict_returns_a_polygon_region(client):
    region = predict(client).json()["results"][0]["result"][0]
    assert region["type"] == "polygonlabels"
    assert len(region["value"]["points"]) >= 3
    for x, y in region["value"]["points"]:
        assert 0 <= x <= 100 and 0 <= y <= 100


def test_predict_output_differs_from_the_baseline(client, routing_task):
    """If the served mask equalled the baseline there would be nothing to correct."""
    region = predict(client).json()["results"][0]["result"][0]
    served_percent = region["value"]["points"]
    baseline_percent = [[x / 800 * 100, y / 600 * 100] for x, y in routing_task.baseline_mask.points]
    assert served_percent != baseline_percent


def test_webhook_endpoint_accepts_project_events(client):
    assert client.post("/webhook", json={"action": "PROJECT_UPDATED"}).status_code == 200


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------

def test_empty_queue_yields_an_empty_prediction_not_an_error(client, monkeypatch):
    def empty(**kwargs):
        raise NoTaskAvailable("junior")

    monkeypatch.setattr(client.app.state.task_source, "next_task", empty)
    response = predict(client)
    assert response.status_code == 200
    assert response.json()["results"][0]["result"] == []


def test_upstream_failure_yields_an_empty_prediction_not_an_error(client, monkeypatch):
    def boom(**kwargs):
        raise TaskSourceError("routing_qa unreachable")

    monkeypatch.setattr(client.app.state.task_source, "next_task", boom)
    response = predict(client)
    assert response.status_code == 200
    assert response.json()["results"][0]["result"] == []


def test_degenerate_baseline_yields_an_empty_prediction_not_an_error(client, monkeypatch, routing_task):
    broken = routing_task.model_copy(
        update={"baseline_mask": routing_task.baseline_mask.model_copy(
            update={"points": [[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]]}
        )}
    )
    monkeypatch.setattr(client.app.state.task_source, "next_task", lambda **kw: broken)
    response = predict(client)
    assert response.status_code == 200
    assert response.json()["results"][0]["result"] == []


def test_predict_survives_a_task_with_no_data(client):
    assert predict(client, tasks=[{"id": 1}]).status_code == 200


def test_unreliable_image_dimensions_suppress_the_pre_annotation(client, monkeypatch):
    """
    A wrong-scale mask is worse than no mask: the annotator corrects it anyway
    and the telemetry then describes a polygon the policy never proposed. Losing
    one rollout is the cheaper failure.
    """
    from app.image_meta import ImageDims

    monkeypatch.setattr(
        client.app.state.dimensions, "resolve",
        lambda *a, **kw: ImageDims(1920, 1080, "fixed", False),
    )
    response = predict(client)
    assert response.status_code == 200
    assert response.json()["results"][0]["result"] == []


def test_reliable_dimensions_still_serve(client, monkeypatch):
    from app.image_meta import ImageDims

    monkeypatch.setattr(
        client.app.state.dimensions, "resolve",
        lambda *a, **kw: ImageDims(640, 480, "probe", True),
    )
    region = predict(client).json()["results"][0]["result"][0]
    assert region["original_width"] == 640
    assert region["original_height"] == 480


# ----------------------------------------------------------------------
# Honeypot secrecy — an explicit merge criterion in the plan
# ----------------------------------------------------------------------

def test_ground_truth_mask_never_appears_in_a_prediction(client, monkeypatch, routing_task):
    from common.schemas.routing_queue import HoneypotMeta
    from common.schemas.tier1_ingestion import PolygonMask

    leaked = routing_task.model_copy(update={"honeypot": HoneypotMeta(
        is_honeypot=True,
        ground_truth_mask=PolygonMask(points=[[1.0, 1.0], [2.0, 2.0], [3.0, 1.0]]),
    )})
    monkeypatch.setattr(client.app.state.task_source, "next_task", lambda **kw: leaked)

    body = predict(client).text
    assert "ground_truth_mask" not in body
    assert "is_honeypot" not in body


def test_task_source_strips_a_leaked_ground_truth_mask(settings, routing_task_raw):
    """Defence in depth: Dev 2 is required to strip it, but we do not rely on that."""
    from app.task_source import _parse_task

    routing_task_raw["honeypot"] = {
        "is_honeypot": True,
        "ground_truth_mask": {"points": [[1, 1], [2, 2], [3, 1]]},
    }
    task = _parse_task(routing_task_raw, source="test")
    assert task.honeypot.ground_truth_mask is None
    assert task.honeypot.is_honeypot is True


# ----------------------------------------------------------------------
# Served-wiggle store
# ----------------------------------------------------------------------

def test_served_polygon_is_retrievable_by_task_and_by_seed(client):
    model_version = predict(client).json()["results"][0]["model_version"]
    seed = model_version.split("|seed=")[1]

    by_seed = client.get(f"/served/by-seed/{seed}")
    assert by_seed.status_code == 200

    record = by_seed.json()
    assert record["wiggle_seed"] == seed
    assert len(record["wiggled_points"]) >= 3
    assert record["baseline_points"] != record["wiggled_points"]

    by_task = client.get(f"/served/{record['task_id']}")
    assert by_task.status_code == 200
    assert by_task.json()["wiggle_seed"] == seed


def test_served_lookup_404s_for_an_unknown_task(client):
    assert client.get("/served/task_that_was_never_served").status_code == 404


def test_served_record_is_appended_to_the_jsonl_store(client, settings):
    from pathlib import Path

    predict(client)
    lines = Path(settings.served_store_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["wiggled_points"]


def test_each_serve_gets_a_fresh_seed(client):
    """A requeued task must explore somewhere new, not replay a rejected action."""
    first = predict(client).json()["results"][0]["model_version"]
    second = predict(client).json()["results"][0]["model_version"]
    assert first != second


# ----------------------------------------------------------------------
# Telemetry beacon
# ----------------------------------------------------------------------

def beacon(client, **overrides):
    body = {
        "task_id": "task_a1b2c3",
        "effort_telemetry": {
            "click_count": 12,
            "cursor_path_length_px": 842.5,
            "dwell_time_ms": 5100,
        },
        "completed_by": "annotator_42",
    }
    body.update(overrides)
    # text/plain, as navigator.sendBeacon sends it.
    return client.post("/telemetry/raw", content=json.dumps(body),
                       headers={"Content-Type": "text/plain;charset=UTF-8"})


def test_beacon_is_accepted_as_text_plain(client):
    """sendBeacon uses text/plain to avoid a CORS preflight that would fail silently."""
    response = beacon(client)
    assert response.status_code == 200
    assert response.json()["accepted"] is True


def test_beacon_still_200s_when_the_gateway_is_unreachable(client):
    """The annotator's submit must not be coupled to Dev 4's health."""
    response = beacon(client)
    assert response.status_code == 200
    assert response.json()["forwarded"] is False


def test_malformed_beacon_is_rejected_with_422(client):
    response = client.post("/telemetry/raw", content="{not json",
                           headers={"Content-Type": "text/plain"})
    assert response.status_code == 422


def test_beacon_missing_required_effort_terms_is_rejected(client):
    response = client.post(
        "/telemetry/raw",
        content=json.dumps({"task_id": "t", "effort_telemetry": {"click_count": 3}}),
        headers={"Content-Type": "text/plain"},
    )
    assert response.status_code == 422


def test_missing_seed_is_recovered_from_the_served_store(client):
    """
    The browser cannot always read the seed back out of the prediction. Losing it
    would sever the link between the effort and the action that caused it, so the
    server fills it in from what it actually served.
    """
    model_version = predict(client).json()["results"][0]["model_version"]
    seed = model_version.split("|seed=")[1]

    forwarder = client.app.state.forwarder
    from app.models import RawTelemetryEnvelope

    envelope = RawTelemetryEnvelope.model_validate({
        "task_id": "task_a1b2c3",
        "effort_telemetry": {"click_count": 1, "cursor_path_length_px": 1.0, "dwell_time_ms": 1},
    })
    enriched = forwarder.enrich(envelope)
    assert enriched.wiggle_seed == seed
    assert enriched.effort_telemetry.wiggle_seed == seed


# ----------------------------------------------------------------------
# Calibration endpoint
# ----------------------------------------------------------------------

def test_preview_reports_diagnostics(client):
    body = client.get("/wiggle/preview", params={"sigma": 0.05}).json()
    assert 0.0 <= body["diagnostics"]["iou_vs_baseline"] <= 1.0
    assert body["diagnostics"]["mean_displacement_px"] > 0
    assert len(body["ls_points_percent"]) == len(body["wiggled_points"])


def test_preview_is_reproducible_for_a_given_seed(client):
    params = {"seed": "preview-seed", "sigma": 0.03}
    assert (client.get("/wiggle/preview", params=params).json()["wiggled_points"]
            == client.get("/wiggle/preview", params=params).json()["wiggled_points"])


def test_preview_does_not_write_to_the_served_store(client, settings):
    """A preview is not a serve; recording it would put a polygon no human saw into Tier 3's data."""
    from pathlib import Path

    client.get("/wiggle/preview")
    store = Path(settings.served_store_path)
    assert not store.exists() or store.read_text(encoding="utf-8").strip() == ""


def test_larger_sigma_lowers_iou_in_the_preview(client):
    small = client.get("/wiggle/preview", params={"sigma": 0.01, "seed": "s"}).json()
    large = client.get("/wiggle/preview", params={"sigma": 0.15, "seed": "s"}).json()
    assert large["diagnostics"]["iou_vs_baseline"] < small["diagnostics"]["iou_vs_baseline"]
