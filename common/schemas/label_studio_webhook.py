"""
Tier 2 — Label Studio ANNOTATION_UPDATED Webhook Contract
Owner: Developer 3 (Interactive Serving & Stochastic Policy) defines what gets
       sent — because Label Studio's STOCK webhook does NOT include click
       counts / cursor path / dwell time. Dev 3 must instrument the Label
       Studio Frontend (LSF) with custom event listeners and attach the
       telemetry as `meta` on the annotation result before Label Studio
       fires the webhook. This file is the frozen contract for that meta
       block, plus the outer envelope Dev 4's gateway expects.
Consumed by: Developer 4 (Webhook Gateway) — validates and republishes onto Redis.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, conint, confloat


class LSAction(str, Enum):
    ANNOTATION_CREATED = "ANNOTATION_CREATED"
    ANNOTATION_UPDATED = "ANNOTATION_UPDATED"


class LSResultRegion(BaseModel):
    """One region in Label Studio's `annotation.result` array (polygon/keypoint)."""
    id: str
    type: str = Field(..., description="e.g. 'polygonlabels'")
    value: Dict[str, Any] = Field(..., description="Raw LS region value (points, labels, etc.)")


class LSTelemetryMeta(BaseModel):
    """
    CUSTOM field — not part of stock Label Studio. Injected client-side by
    Dev 3's LSF instrumentation into `annotation.meta.effort_telemetry`
    before submit fires.
    """
    click_count: conint(ge=0) = Field(..., description="C — total clicks/vertex placements")
    cursor_path_length_px: confloat(ge=0) = Field(..., description="L_path — summed Euclidean cursor travel")
    dwell_time_ms: conint(ge=0) = Field(..., description="T_dwell — active hover time over the boundary")
    wiggle_seed: Optional[str] = Field(None, description="RNG seed used for this task's Gaussian wiggle, for reproducibility/debug")


class LSAnnotationUpdatedPayload(BaseModel):
    action: LSAction
    task_id: str = Field(..., description="Our internal QueueTask.task_id, passed via LS task.data")
    annotation_id: str
    project_id: str
    completed_by: str = Field(..., description="annotator_id")
    result: List[LSResultRegion]
    effort_telemetry: LSTelemetryMeta
    lead_time: float = Field(..., description="Seconds from task open to submit, native LS field")
    created_at: str
    updated_at: str
