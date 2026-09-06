"""
Tier 3 / Tier 4 — RLHF Reward & Rollout Contracts
Owner: Tier 3/4 team. NEW module — does not modify tier1_ingestion.py,
routing_queue.py, label_studio_webhook.py, or redis_event.py.

Tier 3 consumes RedisEventEnvelope[LSAnnotationUpdatedPayload] from
`telemetry:ingest` (frozen, imported not redefined) and produces the types
below. Tier 4 consumes ExperienceTuple.

---------------------------------------------------------------------------
ADDITIONS MADE BY DEV 2 (Phase 2). Everything below the original plan text is
additive — no field was removed or retyped, so code written against the plan's
version of this file still constructs. Each addition is justified inline and
in `docs/tier3-dev2-decisions.md`. The plan explicitly permits new fields on
Tier 3-owned schemas (§4: "a new field on a schema Tier 3 owns, so it's safe,
not a frozen-schema violation").

  1. `WiggleCacheEntry.image_width` / `.image_height`  -- REQUIRED.
     Without these delta-IoU is not computable at all; see the block comment on
     WiggleCacheEntry. This is the single most important thing for whoever
     wires up the Tier 2 cache writer to read.
  2. `WiggleCacheEntry.model_version` -- REQUIRED. Closes the "KNOWN GAP" the
     plan records on `Stage1Output.model_version`. `tier3.replay_buffer
     .model_version` is NOT NULL and Tier 4 batches on it, so a tuple without
     one cannot be stored and cannot be trained on.
  3. `WiggleCacheEntry.is_honeypot` / `.m_gold` -- optional, default off. Turns
     the plan's "v2 enhancement" gold-standard path into a switch rather than a
     rewrite. Tier 2 already knows `is_honeypot` (ServedWiggleRecord carries it).
  4. `WiggleCacheEntry.label` -- optional. The plan §4 asks for this so Tier 4
     can log per-batch class diversity.
  5. `Stage1Output.image_width` / `.image_height` -- REQUIRED, carried through
     from the cache entry. Dev 2 cannot recover them from anywhere else.
  6. `RolloutBatchReadyEvent.tuple_ids` uses `min_length`, not `min_items`
     (pydantic v2 spelling; `min_items` is a no-op/deprecated in v2 and this
     repo pins pydantic 2.x everywhere).
---------------------------------------------------------------------------
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
#
# WHY THE DIMENSIONS ARE REQUIRED (Dev 2, verified against the merged code):
#
#   Label Studio returns polygon vertices as PERCENTAGES of the image, 0-100
#   (`result[].value.points`). `m_initial` / `m_wiggled` here are ABSOLUTE
#   PIXELS. IoU between a percent polygon and a pixel polygon is meaningless.
#
#   The percent->pixel map is anisotropic — diag(W/100, H/100) — so it cannot
#   be cancelled out or guessed. IoU *is* invariant under it (both intersection
#   and union areas scale by the same determinant), which is what makes the
#   comparison exact once both polygons are in one frame, but getting them into
#   one frame needs the real W and H.
#
#   Label Studio does put `original_width` / `original_height` on every region,
#   and Tier 2's `ls_format.build_prediction` sets them. They do NOT survive to
#   Tier 3: `services/webhook_gateway/main.py` re-parses the body through
#   `LSAnnotationUpdatedPayload`, whose `LSResultRegion` declares only
#   `id`/`type`/`value`. Pydantic v2 defaults to `extra="ignore"`, so
#   `original_width`, `original_height` and `region.meta` are dropped before
#   the envelope is ever pushed onto `telemetry:ingest`.
#
#   That leaves this cache entry as the only channel. Tier 2 already has the
#   values on hand — `ServedWiggleRecord` (services/serving_ui/app/models.py)
#   carries `image_width`, `image_height`, `model_version`, `is_honeypot` and
#   the label already. The cache writer is a projection of a record that
#   already exists; it does not need to compute anything new.
# ---------------------------------------------------------------------------
class WiggleCacheEntry(BaseModel):
    wiggle_seed: str
    task_id: str
    image_id: str
    m_initial: PolygonMask = Field(..., description="Baseline consensus mask from Tier 1, pre-wiggle")
    m_wiggled: PolygonMask = Field(..., description="The Gaussian-perturbed mask actually shown to the annotator")
    served_at: str = Field(..., description="ISO-8601 UTC, used for TTL sanity checks")

    # --- Dev 2 additions -------------------------------------------------
    image_width: conint(gt=0) = Field(
        ..., description="Pixel width of the image the polygons are expressed in. REQUIRED: "
                         "Label Studio returns M_final in percent and this is the only "
                         "surviving channel for the conversion. See the block comment above."
    )
    image_height: conint(gt=0) = Field(
        ..., description="Pixel height of the image. REQUIRED, same reason as image_width."
    )
    model_version: str = Field(
        ..., description="The SAM2/LoRA checkpoint that produced m_wiggled. REQUIRED: "
                         "tier3.replay_buffer.model_version is NOT NULL and Tier 4 refuses "
                         "to mix versions inside one PPO batch."
    )
    is_honeypot: bool = Field(
        False, description="True when a gold mask exists for this task, which enables the "
                           "strict ΔIoU form instead of the production fallback."
    )
    m_gold: Optional[PolygonMask] = Field(
        None, description="Gold-standard ground truth. Present only for honeypot tasks."
    )
    label: Optional[str] = Field(
        None, description="Class label of the served region, for Tier 4's per-batch "
                          "diversity logging (plan §4)."
    )


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


class EffortWeights(BaseModel):
    w1_clicks: float = Field(1.0, description="Weight on click_count")
    w2_path: float = Field(0.01, description="Weight on cursor_path_length_px (px is high-magnitude, keep small)")
    w3_dwell: float = Field(0.001, description="Weight on dwell_time_ms (ms is high-magnitude, keep small)")


class RawEffortScore(BaseModel):
    """Output of the Biometric Effort Engine: Delta_E = w1*C + w2*L_path + w3*T_dwell."""
    delta_e_raw: float
    weights: EffortWeights


class NormalizedEffortScore(BaseModel):
    """Output of Z-Score Normalization & Sanity Filter."""
    delta_e_raw: float
    delta_e_norm: float = Field(..., description="Z-score normalized against rolling population stats")
    dropped_as_bot: bool = Field(False, description="True if cursor velocity/pattern looked non-human; task is excluded from reward calc")
    population_mean: float
    population_stddev: float


class GeometricDelta(BaseModel):
    """
    Output of the Geometric Delta Engine: Delta_IoU = IoU(M_final) - IoU(M_initial).

    Dev 2 note on how the two IoU terms are filled when no gold mask exists.
    The canonical production fallback (docs/reference/equations.md, from
    tier3.docx) is a single term:

        ΔIoU = 1 - IoU(M_reference, M_final)

    which is exactly this schema's two-term form once M_final is taken as the
    ground-truth proxy — a human looked at it and signed off, so it is the best
    available estimate of truth for that image:

        iou_initial = IoU(M_reference, M_final)      # what the policy proposed
        iou_final   = IoU(M_final, M_final) = 1.0    # the proxy against itself
        delta_iou   = iou_final - iou_initial = 1 - IoU(M_reference, M_final)

    So the schema and the canonical formula agree exactly; neither had to be
    edited. On honeypot tasks `m_gold` replaces the proxy and both terms carry
    independent information. Same code path, different reference — see
    `services/tier3_worker/reward.py`.
    """
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

    Naming note (Dev 2): `docs/authority.md` C1 rejects the "Offline Replay
    Buffer" by name — PPO is on-policy and training on tuples from superseded
    weights breaks the importance-sampling ratio. What this table actually
    implements is the ratified design: single `model_version` per batch,
    consume-once, no random resampling. The behaviour is correct; only the name
    is inherited from the rejected design. Kept as-is so it matches the DDL and
    the plan, and recorded in the decisions doc so the next reader is not
    misled into thinking C1 was reopened.
    """
    tuple_id: str = Field(..., description="UUID4, primary key in replay buffer")
    wiggle_seed: str = Field(..., description="Join key back to the original serve event")
    task_id: str
    annotation_id: str

    state_s_t: PolygonMask = Field(..., description="M_initial — the state the policy acted on")
    action_a_t: PolygonMask = Field(..., description="M_wiggled — the low-dimensional action the policy took")
    reward_r_t: float

    model_version: str = Field(..., description="Which SAM2/LoRA checkpoint produced action_a_t — required for on-policy validity checks")
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

    DEV 1 PLEASE READ — two required fields were added, both copied straight
    off the `WiggleCacheEntry` you already fetch:

        Stage1Output(..., image_width=entry.image_width,
                          image_height=entry.image_height)

    They are required rather than optional on purpose. Dev 2 cannot compute
    delta-IoU without them and cannot recover them from any other source, so a
    Stage1Output lacking them is not partially useful, it is inert. A pydantic
    ValidationError naming `image_width` at construction time is a two-second
    fix; the alternative is a worker that consumes the queue and silently
    produces no rewards at all.
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

    model_version: Optional[str] = Field(
        None, description="Populated from WiggleCacheEntry.model_version, which is now a required field. "
                          "Left Optional so the plan's original signature still validates, but Dev 2 drops "
                          "the tuple when it is None: tier3.replay_buffer.model_version is NOT NULL."
    )

    # --- Dev 2 additions -------------------------------------------------
    image_width: conint(gt=0) = Field(
        ..., description="Copied from WiggleCacheEntry.image_width. Required — see the class docstring."
    )
    image_height: conint(gt=0) = Field(
        ..., description="Copied from WiggleCacheEntry.image_height. Required — see the class docstring."
    )
    is_honeypot: bool = Field(
        False, description="Copied from WiggleCacheEntry.is_honeypot; selects the strict ΔIoU form."
    )
    m_gold: Optional[PolygonMask] = Field(
        None, description="Copied from WiggleCacheEntry.m_gold; present only on honeypot tasks."
    )
    label: Optional[str] = Field(
        None, description="Copied from WiggleCacheEntry.label, for Tier 4 diversity logging."
    )


class RolloutBatchReadyEvent(BaseModel):
    """Published by Tier 3 (or polled for by Tier 4) once N experience tuples of the SAME model_version are staged."""
    batch_id: str
    model_version: str
    tuple_ids: List[str] = Field(..., min_length=1)
    batch_size: int
    ready_at: str
