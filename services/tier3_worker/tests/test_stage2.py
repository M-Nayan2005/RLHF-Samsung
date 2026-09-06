"""
The `process_stage1` entrypoint end to end, against an in-memory buffer.

Two properties matter more than any individual assertion:

  1. A bad annotation is dropped, not raised. Dev 1's consumer loop must keep
     running; one unusable polygon cannot stop the queue.
  2. A broken database *is* raised. That failure is not a property of this
     tuple and will apply to every one after it, so swallowing it would turn an
     outage into invisible data loss.
"""
from __future__ import annotations

import dataclasses

import pytest

from tier3_worker import stage2
from tier3_worker.buffer import InMemoryReplayBuffer

from .conftest import BASELINE, WIGGLED, ls_region, rect_percent
from .test_reward import EXPECTED_DELTA_IOU


@pytest.fixture
def buffer(settings):
    """Wire the stage to an in-memory buffer and reset it between tests."""
    writer = InMemoryReplayBuffer()
    stage2.configure(settings=settings, writer=writer)
    yield writer
    stage2._settings = None
    stage2._writer = None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

async def test_one_annotation_becomes_one_row(buffer, make_stage1):
    await stage2.process_stage1(make_stage1())

    assert len(buffer.rows) == 1
    row = buffer.rows[0]
    assert row["task_id"] == "task_a1b2c3"
    assert row["annotation_id"] == "ann_9f8e7d"
    assert row["model_version"] == "serving-ui-stochastic-0.1.0"
    assert row["consumed_by_ppo"] is False
    assert row["delta_iou"] == pytest.approx(EXPECTED_DELTA_IOU, abs=1e-12)
    assert row["reward_r_t"] == pytest.approx(EXPECTED_DELTA_IOU - 0.3 * 0.5, abs=1e-12)


async def test_state_and_action_are_stored_in_pixels_not_the_unit_frame(buffer, make_stage1):
    """
    Tier 4 has to reconstruct the action in the space the policy emitted it. A
    unit-frame polygon stripped of its image dimensions cannot be scaled back,
    so the buffer keeps the original absolute-pixel rings.
    """
    await stage2.process_stage1(make_stage1())

    row = buffer.rows[0]
    assert row["state_s_t"]["points"] == BASELINE
    assert row["action_a_t"]["points"] == WIGGLED


async def test_the_weights_are_recorded_alongside_the_reward(buffer, make_stage1):
    await stage2.process_stage1(make_stage1())
    row = buffer.rows[0]
    assert (row["alpha"], row["beta"]) == (1.0, 0.3)
    assert row["delta_e_norm"] == 0.5


async def test_label_and_honeypot_flag_reach_the_row(buffer, make_stage1):
    """Tier 4 logs per-batch class diversity and must be able to spot gold tuples."""
    await stage2.process_stage1(make_stage1(is_honeypot=True, m_gold=BASELINE, label="bus"))
    row = buffer.rows[0]
    assert row["label"] == "bus"
    assert row["is_honeypot"] is True


# ---------------------------------------------------------------------------
# Drops — logged, never raised
# ---------------------------------------------------------------------------

async def test_bot_flagged_effort_is_excluded(buffer, make_stage1):
    await stage2.process_stage1(make_stage1(dropped_as_bot=True))
    assert buffer.rows == []


async def test_bot_flag_on_the_effort_score_alone_is_honoured(buffer, make_stage1):
    """
    `Stage1Output.dropped_as_bot` mirrors `effort.dropped_as_bot` for
    convenience. If the two ever disagree, the exclusion wins — a suspected bot
    must not reach the buffer because a mirror field went stale.
    """
    stage1 = make_stage1()
    stage1.effort.dropped_as_bot = True
    await stage2.process_stage1(stage1)
    assert buffer.rows == []


async def test_missing_model_version_is_dropped(buffer, make_stage1):
    """`tier3.replay_buffer.model_version` is NOT NULL and Tier 4 batches on it."""
    await stage2.process_stage1(make_stage1(model_version=None))
    assert buffer.rows == []


async def test_degenerate_polygon_is_dropped_without_raising(buffer, make_stage1):
    await stage2.process_stage1(make_stage1(m_wiggled=[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    assert buffer.rows == []


async def test_annotation_with_no_regions_is_dropped_without_raising(buffer, make_stage1):
    await stage2.process_stage1(make_stage1(ls_result=[]))
    assert buffer.rows == []


async def test_one_bad_annotation_does_not_stop_the_next(buffer, make_stage1):
    """The queue keeps moving. This is the acceptance criterion Dev 1 depends on."""
    await stage2.process_stage1(make_stage1(annotation_id="bad", ls_result=[]))
    await stage2.process_stage1(make_stage1(annotation_id="good"))

    assert [r["annotation_id"] for r in buffer.rows] == ["good"]


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

async def test_replaying_the_same_annotation_writes_one_row(buffer, make_stage1):
    """
    Acceptance criterion: "replay the same annotation_id twice, confirm only one
    replay_buffer row". The UNIQUE constraint is what enforces it — Dev 1's
    sequence ledger runs in a separate transaction, so two workers racing a
    redelivery can both clear it.
    """
    await stage2.process_stage1(make_stage1())
    await stage2.process_stage1(make_stage1())

    assert len(buffer.rows) == 1


async def test_different_annotations_on_one_task_both_land(buffer, make_stage1):
    """A correction-of-a-correction is a distinct annotation, not a duplicate."""
    await stage2.process_stage1(make_stage1(annotation_id="ann_1"))
    await stage2.process_stage1(make_stage1(annotation_id="ann_2"))

    assert len(buffer.rows) == 2


# ---------------------------------------------------------------------------
# Infrastructure failure
# ---------------------------------------------------------------------------

class ExplodingBuffer:
    async def write(self, *_args, **_kwargs):
        raise ConnectionError("postgres is down")


async def test_a_write_failure_is_raised_not_swallowed(settings, make_stage1):
    stage2.configure(settings=settings, writer=ExplodingBuffer())
    try:
        with pytest.raises(ConnectionError):
            await stage2.process_stage1(make_stage1())
    finally:
        stage2._settings = None
        stage2._writer = None


async def test_write_failures_can_be_made_non_fatal(settings, make_stage1, monkeypatch):
    """
    The operational preference may turn out to be the opposite — keep consuming
    and lose the rows. It is a setting, not a rewrite.
    """
    monkeypatch.setenv("TIER3_RAISE_ON_WRITE_FAILURE", "false")
    stage2.configure(settings=settings, writer=ExplodingBuffer())
    try:
        await stage2.process_stage1(make_stage1())  # must not raise
    finally:
        stage2._settings = None
        stage2._writer = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

async def test_reference_mask_setting_reaches_the_stored_reward(make_stage1, settings):
    """
    Q11 flipped: measuring against the consensus mask instead of the action.
    M_final equals M_initial in the fixture, so the stored reward loses its
    geometric term entirely and becomes pure effort penalty.
    """
    writer = InMemoryReplayBuffer()
    stage2.configure(settings=dataclasses.replace(settings, reference_mask="initial"), writer=writer)
    try:
        await stage2.process_stage1(make_stage1())
        assert writer.rows[0]["delta_iou"] == pytest.approx(0.0, abs=1e-9)
        assert writer.rows[0]["reward_r_t"] == pytest.approx(-0.15, abs=1e-9)
    finally:
        stage2._settings = None
        stage2._writer = None
