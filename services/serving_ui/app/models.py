"""
Models local to serving_ui.

**Nothing in this file is a frozen contract.** The four frozen contracts live in
`common/schemas/` and this service imports them read-only; the plan's rule is
that a schema change needs its own PR reviewed by the downstream consumer, and
Dev 3 is not going to unilaterally edit a file Dev 4 validates against.

What is here instead:

* the Label Studio ML Backend wire shapes, which are Label Studio's contract and
  not ours to define;
* `RawTelemetryEnvelope`, the beacon payload this service posts to Dev 4's
  `/telemetry/raw`. The plan says that endpoint accepts "the same
  effort_telemetry shape directly", but effort telemetry on its own carries no
  key to join it to an annotation, so this wraps `LSTelemetryMeta` in the
  identifiers Dev 4 needs. Proposed as an additive contract - see Q15 / D13 and
  `docs/integration-notes.md`;
* `ServedWiggleRecord`, this service's own persistence row for the polygon it
  actually served. Not a shared contract, but the thing that closes the Q11 /
  D10 gap, so its shape is documented as if it were one.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from common.schemas.label_studio_webhook import LSTelemetryMeta


# --------------------------------------------------------------------------
# Label Studio ML Backend protocol
# --------------------------------------------------------------------------

class LSTask(BaseModel):
    """
    One task as Label Studio hands it to an ML backend.

    Permissive on purpose: Label Studio adds and renames envelope fields between
    minor versions, and a strict model here would turn a cosmetic upstream change
    into a hard 422 that blanks the annotator's canvas. We only require `data`.
    """
    model_config = ConfigDict(extra="allow")

    id: Optional[int] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
    project: Optional[int] = None


class LSPredictRequest(BaseModel):
    """Body of the POST /predict that Label Studio makes when a task opens."""
    model_config = ConfigDict(extra="allow")

    tasks: List[LSTask] = Field(default_factory=list)
    project: Optional[str] = None
    label_config: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class LSPredictionResult(BaseModel):
    """One prediction (one task's worth of regions)."""
    model_config = ConfigDict(extra="allow")

    result: List[Dict[str, Any]]
    score: float = 0.0
    model_version: str


class LSPredictResponse(BaseModel):
    """
    What Label Studio expects back.

    One entry in `results` per task in the request, in the same order - Label
    Studio matches them positionally, not by id.
    """
    model_config = ConfigDict(extra="allow")

    results: List[LSPredictionResult] = Field(default_factory=list)
    model_version: Optional[str] = None


class LSSetupRequest(BaseModel):
    """Label Studio calls POST /setup when the backend is connected or a project saves."""
    model_config = ConfigDict(extra="allow")

    project: Optional[str] = None
    schema_: Optional[str] = Field(None, alias="schema")
    label_config: Optional[str] = None
    extra_params: Dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------
# Telemetry beacon (serving_ui -> webhook_gateway)
# --------------------------------------------------------------------------

class RawTelemetryEnvelope(BaseModel):
    """
    Effort telemetry plus the identifiers needed to join it to an annotation.

    Why the identifiers are needed: the beacon fires from the browser around the
    moment of submit, and Label Studio does not mint `annotation_id` until the
    submit round-trip completes. The instrumentation reads the id off the submit
    response when it can, but on "submit and next" the page often tears down
    first, so `annotation_id` is best-effort rather than guaranteed.

    Dev 4 therefore joins the beacon to the stock `ANNOTATION_UPDATED` webhook -
    which carries `annotation_id` and `result` but no telemetry - using, in
    order of preference:

        1. `wiggle_seed`  - unique per served task, minted by this service, and
                            round-tripped through the prediction. The strongest
                            key, and the reason `wiggle_seed` is duplicated at
                            the top level here rather than left inside
                            `effort_telemetry`.
        2. `task_id`      - our `QueueTask.task_id`, stable but reused across a
                            consensus requeue, so pair it with the newest
                            unmatched beacon.
        3. `client_session_id` - per-browser-tab, for disambiguating two
                            annotators open on the same task.

    When `annotation_id` IS present the join is exact. A beacon carrying
    `supersedes_prior_beacon=True` is the same annotation re-sent once the id
    became known: match it on `wiggle_seed` and replace the earlier record.

    Not a frozen contract. Proposed additively - see `docs/integration-notes.md`.
    """
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(..., description="Our QueueTask.task_id, from LS task.data.task_id")
    effort_telemetry: LSTelemetryMeta

    annotation_id: Optional[str] = Field(
        None, description="Known only when editing an existing annotation; null on first submit"
    )
    project_id: Optional[str] = None
    completed_by: Optional[str] = Field(None, description="annotator_id")
    ls_task_id: Optional[int] = Field(None, description="Label Studio's own numeric task id")
    client_session_id: Optional[str] = Field(
        None, description="Random per-tab id, disambiguates concurrent annotators on one task"
    )
    wiggle_seed: Optional[str] = Field(
        None, description="Lifted from effort_telemetry for join convenience; same value"
    )
    lead_time: Optional[float] = Field(
        None, description="Seconds from task open to submit, measured client-side"
    )
    cursor_path_length_image_px: Optional[float] = Field(
        None,
        description=(
            "The same cursor travel, re-expressed in image-pixel space. Diagnostic only. "
            "`effort_telemetry.cursor_path_length_px` carries CSS pixels - physical cursor "
            "travel, which is what 'effort' means - but that makes L_path depend on the "
            "annotator's zoom level. Recording both means the choice can be settled "
            "empirically later without re-instrumenting the frontend. See Q16 / DD-6."
        ),
    )
    client_sent_at: Optional[str] = Field(None, description="ISO-8601 UTC, browser clock")
    transport: str = Field(
        "beacon", description="beacon | ls_meta - which capture path produced this record"
    )
    supersedes_prior_beacon: bool = Field(
        False,
        description=(
            "True when this beacon is re-sending an annotation whose telemetry already "
            "went out without an annotation_id. Same wiggle_seed as the earlier record; "
            "treat it as a replacement, not a second annotation. See docs/integration-notes.md."
        ),
    )


class TelemetryAck(BaseModel):
    """Response to a beacon. Deliberately tiny - the browser is mid-submit."""
    accepted: bool
    task_id: str
    forwarded: bool = Field(..., description="Whether the gateway forward succeeded")
    detail: Optional[str] = None


# --------------------------------------------------------------------------
# Served-wiggle persistence
# --------------------------------------------------------------------------

class ServedWiggleRecord(BaseModel):
    """
    The polygon this service actually put in front of a human.

    In RL terms this is the sampled action `A_t`. The frozen contracts persist
    only `wiggle_seed`, which makes `A_t` *reconstructible* but not *stored* -
    and reconstruction holds only while the RNG, the vertex ordering and the
    transform all stay pinned (see `wiggle.py`). Tier 3's `delta-IoU` has to be
    computed against the mask the human corrected, so storing it outright turns
    a three-invariant assumption into a lookup.

    Divergence D15 / question Q11. Nothing consumes this yet; Tier 3 does not
    exist. It costs one JSONL append per served task to keep the option open.
    """
    model_config = ConfigDict(extra="forbid")

    task_id: str
    image_id: str
    wiggle_seed: str
    served_at: str = Field(..., description="ISO-8601 UTC")

    baseline_points: List[List[float]] = Field(
        ..., description="Canonicalised mu - QueueTask.baseline_mask after geometry.canonicalize"
    )
    wiggled_points: List[List[float]] = Field(
        ..., description="The sampled action A_t, absolute pixels"
    )

    wiggle_params: Dict[str, Any]
    action: Dict[str, float]
    diagnostics: Dict[str, float]

    image_width: int
    image_height: int
    image_dims_source: str = Field(
        ..., description="probe | fixed | task_data - how width/height were determined"
    )

    annotator_id: Optional[str] = None
    ls_project_id: Optional[str] = None
    ls_task_id: Optional[int] = None
    model_version: str
    queue: Optional[str] = None
    is_honeypot: bool = False


class WigglePreview(BaseModel):
    """Response of the GET /wiggle/preview debug endpoint."""
    model_config = ConfigDict(extra="forbid")

    baseline_points: List[List[float]]
    wiggled_points: List[List[float]]
    ls_points_percent: List[List[float]]
    params: Dict[str, Any]
    action: Dict[str, float]
    diagnostics: Dict[str, float]
    image_width: int
    image_height: int
