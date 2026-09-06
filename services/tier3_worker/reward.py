"""
The Geometric Delta Engine and the E-DRDE Scalar Evaluator.

Both formulas come from `docs/reference/equations.md`, which is the only
trustworthy source for them — the `.docx` extracts flatten OMML math into
unreadable runs like `IoUMinitial,Mfinal`, so equations are never taken from an
extract.

    R_t   = alpha * ΔIoU - beta * ΔE_norm            (E-DRDE, authority.md C2)

    ΔIoU  = IoU(M_final, M_gold) - IoU(M_ref, M_gold)     when gold exists
    ΔIoU  = 1 - IoU(M_ref, M_final)                       production fallback

**Which mask is `M_ref`.** Open question Q11 / divergence D10: the frozen
schemas label the Tier 1 consensus mask `M_initial`, but the mask the human
actually corrected is the wiggled one, and in RL terms that wiggled polygon is
the sampled action `A_t`. Reward must attribute to the action taken, not to the
mean of the distribution it was sampled from — otherwise every action on a
given task earns an identical reward and the critic has nothing to separate.
So the default reference is `m_wiggled`, overridable with
`TIER3_REFERENCE_MASK` so the pending Q11 decision can be applied without a
code change.

**Why the fallback's sign is not a bug.** `ΔIoU = 1 - IoU(M_ref, M_final)` is
large when the human changed a lot, and it enters `R_t` positively. Read alone
that rewards the policy for proposing bad masks. It is not read alone: a mask
that needed heavy correction also drives `ΔE` up, and `beta * ΔE_norm` subtracts
it. The quantity E-DRDE actually maximises is geometric improvement *per unit
of human effort* — which is the stated purpose of the whole system: "a mask
with 98% initial IoU looks good mathematically, but if that missing 2% required
40 tedious, pinpoint clicks, the AI's initial guess was terrible from an
operational standpoint" (tier3.docx). The balance therefore lives entirely in
alpha and beta, neither of which the source documents ever assign. See
`config.py`.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from common.schemas.tier3_rlhf import EDRDEReward, GeometricDelta, Stage1Output

from .config import Settings
from .geometry import (
    GeometryError,
    exact_iou,
    extract_final_mask,
    pixels_to_unit,
    to_polygon,
)

log = logging.getLogger(__name__)


def compute_geometric_delta(
    stage1: Stage1Output,
    settings: Settings,
) -> Tuple[GeometricDelta, str]:
    """
    Geometric Delta Engine. Returns `(delta, mode)` where mode is 'gold' or
    'fallback', so the caller can record which of the two differently-scaled
    ΔIoU definitions produced this row (open question Q6).

    Raises `GeometryError` if any mask cannot be turned into positive area. The
    caller drops the tuple — a reward computed from a degenerate polygon is
    worse than no reward, because it looks plausible.
    """
    reference_points = (
        stage1.m_wiggled.points if settings.reference_is_wiggled else stage1.m_initial.points
    )
    reference = to_polygon(
        pixels_to_unit(reference_points, stage1.image_width, stage1.image_height)
    )

    final, _points_percent, strategy = extract_final_mask(stage1.ls_result, stage1.label)
    if strategy != "single_region":
        log.info(
            "task=%s M_final assembled from multiple regions via %s",
            stage1.task_id, strategy,
        )

    gold = _gold_geometry(stage1, settings)

    if gold is not None:
        # Strict form: both terms measured against a real ground truth.
        iou_initial = exact_iou(reference, gold)
        iou_final = exact_iou(final, gold)
        mode = "gold"
    else:
        # Production fallback. M_final is the ground-truth proxy — a human
        # looked at it and signed off — which makes `iou_final` its IoU with
        # itself, 1.0, and collapses the two-term form to exactly the canonical
        # single-term fallback `ΔIoU = 1 - IoU(M_ref, M_final)`.
        iou_initial = exact_iou(reference, final)
        iou_final = 1.0
        mode = "fallback"

    return (
        GeometricDelta(
            task_id=stage1.task_id,
            iou_initial=iou_initial,
            iou_final=iou_final,
            delta_iou=iou_final - iou_initial,
        ),
        mode,
    )


def _gold_geometry(stage1: Stage1Output, settings: Settings):
    """The gold mask in the unit frame, or None when the fallback applies."""
    if not settings.use_gold_when_available or stage1.m_gold is None:
        return None
    if not stage1.is_honeypot:
        # A gold mask on a non-honeypot task is a contract inconsistency, not a
        # licence to use it. Prefer the fallback and say so.
        log.warning(
            "task=%s carries m_gold but is_honeypot is False; using the production "
            "fallback ΔIoU rather than trusting an unflagged gold mask",
            stage1.task_id,
        )
        return None
    try:
        return to_polygon(
            pixels_to_unit(stage1.m_gold.points, stage1.image_width, stage1.image_height)
        )
    except GeometryError as exc:
        log.warning(
            "task=%s gold mask is unusable (%s); falling back to the proxy form",
            stage1.task_id, exc,
        )
        return None


def evaluate_edrde(
    stage1: Stage1Output,
    delta: GeometricDelta,
    settings: Settings,
) -> EDRDEReward:
    """
    E-DRDE Scalar Evaluator: `R_t = alpha * ΔIoU - beta * ΔE_norm`.

    `ΔE_norm` arrives already Z-scored by Dev 1. The normalisation is not
    optional and is not re-done here: applying beta to a raw ΔE is the "scale
    disparity" flaw tier3.docx names explicitly — ΔIoU is bounded in [-1, 1]
    while raw ΔE reaches thousands of pixels, so an unnormalised effort term
    obliterates the accuracy term and drives every reward negative.
    """
    r_t = settings.alpha * delta.delta_iou - settings.beta * stage1.effort.delta_e_norm
    return EDRDEReward(
        task_id=stage1.task_id,
        annotation_id=stage1.annotation_id,
        alpha=settings.alpha,
        beta=settings.beta,
        delta_iou=delta.delta_iou,
        delta_e_norm=stage1.effort.delta_e_norm,
        r_t=r_t,
    )
