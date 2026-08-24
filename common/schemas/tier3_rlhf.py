"""
Tier 3 / Tier 4 — RLHF Reward & Rollout Contracts
Owner: Tier 3/4 team. NEW module — does not modify tier1_ingestion.py,
routing_queue.py, label_studio_webhook.py, or redis_event.py.

Tier 3 consumes RedisEventEnvelope[LSAnnotationUpdatedPayload] from
`telemetry:ingest` (frozen, imported not redefined) and produces the types
below. Tier 4 consumes ExperienceTuple.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, confloat, conint

from .tier1_ingestion import PolygonMask


# ---------------------------------------------------------------------------
# Dependency bridge: Tier 2's serving_ui must cache this in REDIS (not
# Postgres) at serve-time, keyed as `wiggle_cache:{wiggle_seed}`, TTL ~24h.
# Tier 3 reads it — never writes it. This is how Tier 3 gets M_initial /
# M_wiggled without touching Tier 1/2's Postgres tables.
# ---------------------------------------------------------------------------
class WiggleCacheEntry(BaseModel):
    wiggle_seed: str
    task_id: str
    image_id: str
    m_initial: PolygonMask = Field(..., description="Baseline consensus mask from Tier 1, pre-wiggle")
    m_wiggled: PolygonMask = Field(..., description="The Gaussian-perturbed mask actually shown to the annotator")
    served_at: str = Field(..., description="ISO-8601 UTC, used for TTL sanity checks")
    model_version: str = Field(..., description="The SAM2 checkpoint version that produced m_initial")
    label: Optional[str] = Field(None, description="[V2] The class label from Tier 1, used for diversity sampling in Tier 4")


class SequenceCheckResult(BaseModel):
    """Output of the Webhook Interceptor & Sequence Check node."""
    annotation_id: str
    task_id: str
    is_duplicate: bool
    is_out_of_order: bool
    accepted: bool = Field(..., description="False => dropped, never reaches the reward math")
    reason: Optional[str] = None


class BiometricSignals(BaseModel):
    """Output of the Custom Telemetry Extractor node — C, L_path, T_dwell pulled straight from effort_telemetry."""
    click_count: conint(ge=0)
    cursor_path_length_px: confloat(ge=0)
    dwell_time_ms: conint(ge=0)


class RawEffortScore(BaseModel):
    """Output of the Biometric Effort Engine: Delta_E = w1*C + w2*L_path + w3*T_dwell."""
    delta_e_raw: float
    weights: "EffortWeights"


class EffortWeights(BaseModel):
    w1_clicks: float = Field(1.0, description="Weight on click_count")
    w2_path: float = Field(0.01, description="Weight on cursor_path_length_px (px is high-magnitude, keep small)")
    w3_dwell: float = Field(0.001, description="Weight on dwell_time_ms (ms is high-magnitude, keep small)")


class NormalizedEffortScore(BaseModel):
    """Output of Z-Score Normalization & Sanity Filter."""
    delta_e_raw: float
    delta_e_norm: float = Field(..., description="Z-score normalized against rolling population stats")
    dropped_as_bot: bool = Field(False, description="True if cursor velocity/pattern looked non-human; task is excluded from reward calc")
    population_mean: float
    population_stddev: float


class GeometricDelta(BaseModel):
    """Output of the Geometric Delta Engine: Delta_IoU = IoU(M_final) - IoU(M_initial)."""
    task_id: str
    iou_initial: confloat(ge=0, le=1) = Field(..., description="IoU(M_initial, ground truth proxy) — see plan doc for how this is estimated without gold labels")
    iou_final: confloat(ge=0, le=1)
    delta_iou: float


class EDRDEReward(BaseModel):
    """Output of the E-DRDE Scalar Evaluator: R_t = alpha*Delta_IoU - beta*Normalized_Delta_E."""
    task_id: str
    annotation_id: str
    alpha: float
    beta: float
    delta_iou: float
    delta_e_norm: float
    r_t: float


class ExperienceTuple(BaseModel):
    """
    Output of the State-Action-Reward Aggregator. This is what gets written
    to the Offline Replay Buffer (Tier 3 -> Tier 4 handoff) and later
    streamed into Tier 4's Rollout Queue.
    """
    tuple_id: str = Field(..., description="UUID4, primary key in replay buffer")
    wiggle_seed: str = Field(..., description="Join key back to the original serve event")
    task_id: str
    annotation_id: str

    state_s_t: PolygonMask = Field(..., description="M_initial — the state the policy acted on")
    action_a_t: PolygonMask = Field(..., description="M_wiggled — the low-dimensional action the policy took")
    reward_r_t: float

    model_version: str = Field(..., description="Which SAM2/LoRA checkpoint produced action_a_t — required for on-policy validity checks")
    label: Optional[str] = Field(None, description="[V2] Inherited from WiggleCacheEntry for per-batch diversity monitoring")
    created_at: str
    consumed_by_ppo: bool = Field(False, description="Flipped true + row deleted/archived once Tier 4 flushes its batch")


class Stage1Output(BaseModel):
    """
    THE Dev1 -> Dev2 HANDOFF CONTRACT. Frozen the same way everything else
    is. Dev 1 constructs this and calls Dev 2's entrypoint function with it
    (in-process, same worker) — Dev 2 consumes ONLY this object, never the
    raw envelope/telemetry directly, so both sides can build against this
    file alone without a live sync.

    Dev 1's entrypoint signature, exactly:
        async def process_envelope(envelope: RedisEventEnvelope) -> Optional[Stage1Output]
        # returns None if dropped (duplicate/out-of-order/missing cache/bot-flagged upstream)

    Dev 2's entrypoint signature, exactly:
        async def process_stage1(stage1: Stage1Output) -> None
        # writes to tier3.replay_buffer, or excludes+logs if stage1.dropped_as_bot
    """
    annotation_id: str
    task_id: str
    wiggle_seed: str

    m_initial: PolygonMask
    m_wiggled: PolygonMask
    ls_result: List[dict] = Field(
        ..., description="Raw payload.result from the frozen LSAnnotationUpdatedPayload — Dev 2 parses M_final out of this, Dev 1 does not parse it"
    )

    effort: NormalizedEffortScore
    dropped_as_bot: bool = Field(False, description="Mirrors effort.dropped_as_bot for convenience — Dev 2 must check this and exclude from replay_buffer")

    model_version: str = Field(
        ..., description="Populated from WiggleCacheEntry. Required to match ExperienceTuple."
    )
    label: Optional[str] = Field(None, description="[V2] Populated from WiggleCacheEntry for diversity monitoring")


class RolloutBatchReadyEvent(BaseModel):
    """Published by Tier 3 (or polled for by Tier 4) once N experience tuples of the SAME model_version are staged."""
    batch_id: str
    model_version: str
    tuple_ids: List[str] = Field(..., min_items=1)
    batch_size: int
    ready_at: str
