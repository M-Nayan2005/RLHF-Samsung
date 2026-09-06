"""
State-Action-Reward Aggregator.

Assembles the `ExperienceTuple` that Tier 4 trains on. The RL reading of the
three slots, spelled out because the naming in the frozen Tier 1/2 schemas
points the other way (open question Q11):

    state  S_t  = M_initial   the consensus mask the policy conditioned on
    action A_t  = M_wiggled   the polygon actually sampled and shown to a human
    reward R_t  = E-DRDE      accuracy gain penalised by the effort it cost

`state_s_t` and `action_a_t` are stored as the original absolute-pixel
polygons, not the normalised unit-frame copies the IoU is computed in. Tier 4
needs to reconstruct the action in the same space the policy emitted it, and a
unit-frame polygon without its image dimensions cannot be scaled back.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from common.schemas.tier3_rlhf import EDRDEReward, ExperienceTuple, Stage1Output


def utc_now_iso() -> str:
    """ISO-8601 UTC with a trailing Z, matching the convention in the frozen schemas."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def build_experience_tuple(stage1: Stage1Output, reward: EDRDEReward) -> ExperienceTuple:
    """
    One row of training data.

    `model_version` is required here and on the table. The caller checks it is
    present before reaching this function — see `stage2.process_stage1` — so a
    None arriving at this point is a programming error rather than a data
    problem, and is allowed to raise.
    """
    if not stage1.model_version:
        raise ValueError(
            "model_version is required to build an ExperienceTuple: Tier 4 batches "
            "on it and mixing checkpoints breaks PPO's importance-sampling ratio"
        )

    return ExperienceTuple(
        tuple_id=str(uuid.uuid4()),
        wiggle_seed=stage1.wiggle_seed,
        task_id=stage1.task_id,
        annotation_id=stage1.annotation_id,
        state_s_t=stage1.m_initial,
        action_a_t=stage1.m_wiggled,
        reward_r_t=reward.r_t,
        model_version=stage1.model_version,
        created_at=utc_now_iso(),
        consumed_by_ppo=False,
    )
