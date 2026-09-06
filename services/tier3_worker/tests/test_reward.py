"""
Geometric Delta Engine and E-DRDE Scalar Evaluator.

The expected numbers are worked out by hand in the comments rather than
captured from a run, so a regression shows up as a disagreement with the
arithmetic instead of a disagreement with whatever the code did last time.
"""
from __future__ import annotations

import dataclasses

import pytest

from tier3_worker.geometry import GeometryError
from tier3_worker.reward import compute_geometric_delta, evaluate_edrde

from .conftest import BASELINE, WIGGLED, ls_region, rect, rect_percent

# Worked by hand, in the unit frame (image 800x400):
#   M_wiggled  x 120/800..420/800 = 0.150..0.525   y 60/400..310/400 = 0.150..0.775
#   M_final    x 100/800..400/800 = 0.125..0.500   y 50/400..300/400 = 0.125..0.750
#   intersect  0.350 x 0.600                       = 0.210000
#   areas      0.375 x 0.625 each                  = 0.234375
#   union      0.234375 + 0.234375 - 0.210000      = 0.258750
#   IoU        0.210000 / 0.258750                 = 0.8115942028985508
EXPECTED_IOU = 0.21 / 0.25875
EXPECTED_DELTA_IOU = 1.0 - EXPECTED_IOU


# ---------------------------------------------------------------------------
# Geometric Delta Engine — production fallback
# ---------------------------------------------------------------------------

def test_fallback_delta_matches_the_canonical_formula(make_stage1, settings):
    """ΔIoU = 1 - IoU(M_ref, M_final), per docs/reference/equations.md."""
    delta, mode = compute_geometric_delta(make_stage1(), settings)

    assert mode == "fallback"
    assert delta.iou_initial == pytest.approx(EXPECTED_IOU, abs=1e-12)
    assert delta.iou_final == 1.0
    assert delta.delta_iou == pytest.approx(EXPECTED_DELTA_IOU, abs=1e-12)


def test_untouched_mask_scores_zero_improvement(make_stage1, settings):
    """
    The human accepted the served polygon unchanged. Nothing was corrected, so
    the geometric term contributes nothing and the reward is pure effort
    penalty.
    """
    stage1 = make_stage1(ls_result=[ls_region(rect_percent(120, 60, 420, 310))])
    delta, _mode = compute_geometric_delta(stage1, settings)

    assert delta.iou_initial == pytest.approx(1.0, abs=1e-9)
    assert delta.delta_iou == pytest.approx(0.0, abs=1e-9)


def test_heavily_corrected_mask_scores_near_one(make_stage1, settings):
    """A human who replaced the mask entirely leaves ΔIoU at its ceiling."""
    stage1 = make_stage1(ls_result=[ls_region(rect_percent(600, 300, 780, 390))])
    delta, _mode = compute_geometric_delta(stage1, settings)

    assert delta.iou_initial == 0.0
    assert delta.delta_iou == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Q11: which mask is the reference
# ---------------------------------------------------------------------------

def test_reference_defaults_to_the_wiggled_mask(settings):
    """
    The default has to be the action the policy actually took. Reward attributed
    to the distribution mean instead of the sample gives every action on a task
    an identical score and leaves the critic nothing to separate.
    """
    assert settings.reference_is_wiggled


def test_reference_mask_switch_changes_the_answer(make_stage1, settings):
    """
    Q11 / D10 is still unresolved, so the choice is a setting rather than a
    hardcoded decision. Here M_final equals M_initial exactly, so measuring
    against the consensus mask reports a perfect zero-improvement annotation
    while measuring against the action reports the real correction.
    """
    stage1 = make_stage1()
    against_initial = dataclasses.replace(settings, reference_mask="initial")

    delta_wiggled, _ = compute_geometric_delta(stage1, settings)
    delta_initial, _ = compute_geometric_delta(stage1, against_initial)

    assert delta_wiggled.delta_iou == pytest.approx(EXPECTED_DELTA_IOU, abs=1e-12)
    assert delta_initial.delta_iou == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Gold-standard path (honeypots)
# ---------------------------------------------------------------------------

def test_gold_path_uses_both_terms_independently(make_stage1, settings):
    """
    With a real ground truth the two IoU terms carry independent information and
    ΔIoU is the strict form, IoU(M_final, M_gold) - IoU(M_ref, M_gold).
    """
    stage1 = make_stage1(is_honeypot=True, m_gold=BASELINE)
    delta, mode = compute_geometric_delta(stage1, settings)

    assert mode == "gold"
    # M_final == M_gold == BASELINE here, so the final term is a perfect 1.0 and
    # the initial term is the wiggled mask's agreement with the gold mask.
    assert delta.iou_final == pytest.approx(1.0, abs=1e-9)
    assert delta.iou_initial == pytest.approx(EXPECTED_IOU, abs=1e-12)
    assert delta.delta_iou == pytest.approx(EXPECTED_DELTA_IOU, abs=1e-12)


def test_gold_mask_on_a_non_honeypot_task_is_not_trusted(make_stage1, settings):
    """A gold mask without the honeypot flag is a contract inconsistency, not a licence."""
    stage1 = make_stage1(is_honeypot=False, m_gold=BASELINE)
    _delta, mode = compute_geometric_delta(stage1, settings)
    assert mode == "fallback"


def test_gold_path_can_be_switched_off(make_stage1, settings):
    """Q6 asks whether mixing both ΔIoU definitions in one batch is acceptable."""
    stage1 = make_stage1(is_honeypot=True, m_gold=BASELINE)
    no_gold = dataclasses.replace(settings, use_gold_when_available=False)
    _delta, mode = compute_geometric_delta(stage1, no_gold)
    assert mode == "fallback"


def test_unusable_gold_mask_falls_back_rather_than_failing(make_stage1, settings):
    """A broken gold mask must not cost the tuple; the proxy form still works."""
    stage1 = make_stage1(is_honeypot=True, m_gold=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    delta, mode = compute_geometric_delta(stage1, settings)
    assert mode == "fallback"
    assert delta.delta_iou == pytest.approx(EXPECTED_DELTA_IOU, abs=1e-12)


# ---------------------------------------------------------------------------
# Failure modes that must surface as GeometryError, not a plausible number
# ---------------------------------------------------------------------------

def test_annotation_with_no_polygon_regions_raises(make_stage1, settings):
    stage1 = make_stage1(ls_result=[{"id": "r1", "type": "choices", "value": {"choices": ["y"]}}])
    with pytest.raises(GeometryError):
        compute_geometric_delta(stage1, settings)


def test_degenerate_served_mask_raises(make_stage1, settings):
    stage1 = make_stage1(m_wiggled=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    with pytest.raises(GeometryError):
        compute_geometric_delta(stage1, settings)


# ---------------------------------------------------------------------------
# E-DRDE Scalar Evaluator
# ---------------------------------------------------------------------------

def test_edrde_is_alpha_times_iou_minus_beta_times_effort(make_stage1, settings):
    stage1 = make_stage1()
    delta, _mode = compute_geometric_delta(stage1, settings)
    reward = evaluate_edrde(stage1, delta, settings)

    expected = settings.alpha * EXPECTED_DELTA_IOU - settings.beta * 0.5
    assert reward.r_t == pytest.approx(expected, abs=1e-12)
    assert reward.delta_iou == pytest.approx(EXPECTED_DELTA_IOU, abs=1e-12)
    assert reward.delta_e_norm == 0.5
    assert reward.annotation_id == stage1.annotation_id


def test_the_weights_used_are_recorded_on_the_reward(make_stage1, settings):
    """
    alpha and beta are provisional and will be retuned. Every row stores the
    pair it was computed with, so a buffer that spans a retune stays
    interpretable instead of silently mixing two reward scales.
    """
    tuned = dataclasses.replace(settings, alpha=2.0, beta=0.05)
    stage1 = make_stage1()
    delta, _mode = compute_geometric_delta(stage1, tuned)
    reward = evaluate_edrde(stage1, delta, tuned)

    assert (reward.alpha, reward.beta) == (2.0, 0.05)
    assert reward.r_t == pytest.approx(2.0 * EXPECTED_DELTA_IOU - 0.05 * 0.5, abs=1e-12)


def test_high_effort_drives_the_reward_negative(make_stage1, settings):
    """
    The point of E-DRDE: an expensive correction is a bad outcome even when the
    geometry improved. A mask at 98% IoU that cost 40 pinpoint clicks was a bad
    guess operationally, and the effort term is what says so.
    """
    stage1 = make_stage1(delta_e_norm=4.0)
    delta, _mode = compute_geometric_delta(stage1, settings)
    reward = evaluate_edrde(stage1, delta, settings)

    assert reward.r_t < 0
    assert reward.r_t == pytest.approx(EXPECTED_DELTA_IOU - 0.3 * 4.0, abs=1e-12)


def test_effortless_correction_beats_an_identical_one_that_cost_more(make_stage1, settings):
    """Same geometry, different effort: the cheaper annotation must score higher."""
    cheap = make_stage1(delta_e_norm=-1.0)
    dear = make_stage1(delta_e_norm=2.0)

    r_cheap = evaluate_edrde(cheap, compute_geometric_delta(cheap, settings)[0], settings)
    r_dear = evaluate_edrde(dear, compute_geometric_delta(dear, settings)[0], settings)

    assert r_cheap.r_t > r_dear.r_t
