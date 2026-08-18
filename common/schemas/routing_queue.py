"""
Tier 1 — Router Output / Queue Task Contract
Owner: Developer 2 (Routing, QA & Honeypot Engine)
Consumed by: Developer 3 (Label Studio ML Backend pulls tasks from `junior_queue`),
             Senior Annotator Pool UI pulls from `senior_queue`,
             IAA Consensus Engine pulls from `consensus_queue` (stubbed tonight).

This is the record pushed onto Postgres tables `junior_queue` / `senior_queue` /
`consensus_queue`, and mirrored as a Redis list entry (key = queue name) holding
just the task_id for fast polling.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, confloat

from .tier1_ingestion import PolygonMask, BoundingBox


class QueueType(str, Enum):
    JUNIOR = "junior_queue"
    SENIOR = "senior_queue"
    CONSENSUS = "consensus_queue"


class RoutingMetrics(BaseModel):
    geometric_variance: confloat(ge=0)
    class_logit_entropy: confloat(ge=0)
    routed_reason: str = Field(
        ..., description="Human-readable routing rationale, e.g. 'low_entropy_low_variance' or 'high_entropy_confidently_wrong'"
    )


class HoneypotMeta(BaseModel):
    is_honeypot: bool = False
    ground_truth_mask: Optional[PolygonMask] = Field(
        None, description="Only populated when is_honeypot=True. NEVER sent to the frontend — server-side only."
    )


class QueueTask(BaseModel):
    task_id: str = Field(..., description="Stable UUID, distinct from image_id (one image can be re-queued)")
    image_id: str
    image_url: str
    bounding_box: BoundingBox
    baseline_mask: PolygonMask = Field(..., description="== consensus_mask from Tier 1, this is M_initial")

    queue: QueueType
    routing_metrics: RoutingMetrics
    honeypot: HoneypotMeta = Field(default_factory=HoneypotMeta)

    retry_count: int = Field(0, description="Incremented only for consensus_queue re-routes")
    status: str = Field("pending", description="pending | assigned | completed | discarded")
    assigned_to: Optional[str] = Field(None, description="annotator_id once claimed")
    created_at: str
    updated_at: str
