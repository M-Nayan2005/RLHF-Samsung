"""
PPO on the 5-d affine wiggle policy.

The policy is `common/rl_policy.py`'s Gaussian, re-expressed in torch so it can
be differentiated. Each served task is one decision with one reward, so every
rollout is a single-step episode: the return is the reward, and the advantage
is `A = R - V(s)` (no bootstrapping, no gamma). PPO still earns its keep here:
the clipped importance ratio is what lets one batch of 64 human annotations be
reused for several gradient epochs without the policy running away from the
data it was collected under.

Every rollout carries `action_log_prob`, the log-prob the *serving* policy
assigned to its action. Before training, the batch is re-scored under the
checkpoint it claims to come from; a mismatch means the batch is not on-policy
for that checkpoint and the run is refused rather than trained on silently.
"""
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from common import rl_policy

logger = logging.getLogger(__name__)

DTYPE = torch.float64
_LOG_2PI = math.log(2.0 * math.pi)


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass(frozen=True)
class PPOConfig:
    """
    Optimiser settings. Standard PPO defaults (Schulman et al. 2017, and the
    common reference implementations) except `actor_lr`, which is 10x the usual
    3e-4: this policy has ~400 parameters and a batch costs an hour of human
    annotation, so the usual rate barely moves it per batch. `target_kl` is the
    guardrail against that rate overshooting. None of these come from the
    Samsung spec, which sets none. Overridable from the environment; see README.
    """
    epochs: int = 10
    minibatch_size: int = 16
    clip_ratio: float = 0.2
    actor_lr: float = 3e-3
    critic_lr: float = 1e-3
    entropy_coef: float = 1e-3
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    logp_tolerance: float = 1e-4

    @classmethod
    def from_env(cls) -> "PPOConfig":
        return cls(
            epochs=_env_int("PPO_EPOCHS", cls.epochs),
            minibatch_size=_env_int("PPO_MINIBATCH_SIZE", cls.minibatch_size),
            clip_ratio=_env_float("PPO_CLIP_RATIO", cls.clip_ratio),
            actor_lr=_env_float("PPO_ACTOR_LR", cls.actor_lr),
            critic_lr=_env_float("PPO_CRITIC_LR", cls.critic_lr),
            entropy_coef=_env_float("PPO_ENTROPY_COEF", cls.entropy_coef),
            max_grad_norm=_env_float("PPO_MAX_GRAD_NORM", cls.max_grad_norm),
            target_kl=_env_float("PPO_TARGET_KL", cls.target_kl),
        )


class TorchPolicy(nn.Module):
    """Mirror of `rl_policy.PolicyParams`. `tests/test_parity.py` pins the two together."""

    def __init__(self, params: rl_policy.PolicyParams):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(params.w1, dtype=DTYPE))
        self.b1 = nn.Parameter(torch.tensor(params.b1, dtype=DTYPE))
        self.w2 = nn.Parameter(torch.tensor(params.w2, dtype=DTYPE))
        self.b2 = nn.Parameter(torch.tensor(params.b2, dtype=DTYPE))
        self.log_std = nn.Parameter(torch.tensor(params.log_std, dtype=DTYPE))

    def mean(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(obs @ self.w1.T + self.b1)
        limit = rl_policy.MEAN_LIMIT
        return limit * torch.tanh((h @ self.w2.T + self.b2) / limit)

    def clamped_log_std(self) -> torch.Tensor:
        return torch.clamp(self.log_std, rl_policy.LOG_STD_MIN, rl_policy.LOG_STD_MAX)

    def log_prob(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        mu = self.mean(obs)
        log_std = self.clamped_log_std()
        return torch.sum(-0.5 * ((z - mu) / torch.exp(log_std)) ** 2 - log_std - 0.5 * _LOG_2PI, dim=-1)

    def entropy(self) -> torch.Tensor:
        return torch.sum(self.clamped_log_std() + 0.5 * (1.0 + _LOG_2PI))

    def export(self, version: str, parent_version: Optional[str]) -> rl_policy.PolicyParams:
        def arr(t: torch.Tensor) -> np.ndarray:
            return t.detach().cpu().numpy().astype(np.float64)

        return rl_policy.PolicyParams(
            version=version,
            parent_version=parent_version,
            w1=arr(self.w1), b1=arr(self.b1), w2=arr(self.w2), b2=arr(self.b2),
            log_std=arr(self.log_std),
        )


class ValueNetwork(nn.Module):
    """V(s): expected reward of serving the current policy on state s. The PPO baseline."""

    def __init__(self, obs_dim: int = rl_policy.OBS_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        ).to(DTYPE)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class OffPolicyBatchError(ValueError):
    """The batch's recorded log-probs do not match the checkpoint it is tagged with."""


def batch_tensors(rows: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    obs = torch.tensor([r["policy_obs"] for r in rows], dtype=DTYPE)
    z = torch.tensor([r["action_z"] for r in rows], dtype=DTYPE)
    logp_old = torch.tensor([float(r["action_log_prob"]) for r in rows], dtype=DTYPE)
    rewards = torch.tensor([float(r["reward_r_t"]) for r in rows], dtype=DTYPE)
    if obs.shape[1:] != (rl_policy.OBS_DIM,) or z.shape[1:] != (rl_policy.ACTION_DIM,):
        raise ValueError(f"batch has obs {tuple(obs.shape)} / actions {tuple(z.shape)}; expected "
                         f"(n, {rl_policy.OBS_DIM}) / (n, {rl_policy.ACTION_DIM})")
    return obs, z, logp_old, rewards


class PPOTrainer:
    def __init__(self, config: Optional[PPOConfig] = None, seed: Optional[int] = None):
        self.config = config or PPOConfig()
        self.generator = torch.Generator().manual_seed(seed) if seed is not None else None

    def train(
        self,
        params: rl_policy.PolicyParams,
        rows: List[Dict[str, Any]],
        critic_state: Optional[Dict[str, Any]] = None,
    ) -> Tuple[TorchPolicy, ValueNetwork, Dict[str, float]]:
        """
        One PPO update over one batch collected under `params`.

        Returns the updated actor, the updated critic, and the metrics that go
        into `tier3.ppo_training_runs`.
        """
        cfg = self.config
        obs, z, logp_old, rewards = batch_tensors(rows)
        n = obs.shape[0]

        actor = TorchPolicy(params)
        critic = ValueNetwork()
        if critic_state is not None:
            critic.load_state_dict(critic_state)

        with torch.no_grad():
            logp_check = actor.log_prob(obs, z)
            mismatch = float(torch.max(torch.abs(logp_check - logp_old)))
        if mismatch > cfg.logp_tolerance:
            raise OffPolicyBatchError(
                f"recorded log-probs differ from checkpoint {params.version!r} by up to {mismatch:.3g}; "
                "the batch was not served by this policy"
            )

        actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)

        with torch.no_grad():
            value_before = critic(obs)
        critic_loss_before = float(torch.mean((value_before - rewards) ** 2))

        actor_losses, critic_losses, clip_fracs = [], [], []
        approx_kl = 0.0
        epochs_run = 0
        for _ in range(cfg.epochs):
            # The advantage baseline is recomputed each epoch, so the critic
            # learning within the batch sharpens the actor's signal.
            with torch.no_grad():
                adv = rewards - critic(obs)
                if n > 1:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            perm = torch.randperm(n, generator=self.generator)
            for start in range(0, n, cfg.minibatch_size):
                idx = perm[start:start + cfg.minibatch_size]
                logp = actor.log_prob(obs[idx], z[idx])
                ratio = torch.exp(logp - logp_old[idx])
                surr1 = ratio * adv[idx]
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * actor.entropy()

                actor_opt.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
                actor_opt.step()

                critic_loss = torch.mean((critic(obs[idx]) - rewards[idx]) ** 2)
                critic_opt.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
                critic_opt.step()

                actor_losses.append(float(actor_loss))
                critic_losses.append(float(critic_loss))
                clip_fracs.append(float(torch.mean((torch.abs(ratio - 1.0) > cfg.clip_ratio).to(DTYPE))))

            epochs_run += 1
            with torch.no_grad():
                log_ratio = actor.log_prob(obs, z) - logp_old
                # The low-variance KL estimator from Schulman's "Approximating KL" note.
                approx_kl = float(torch.mean(torch.exp(log_ratio) - 1.0 - log_ratio))
            if approx_kl > cfg.target_kl:
                logger.info("Early stop after %d epochs: approx KL %.4f > target %.4f",
                            epochs_run, approx_kl, cfg.target_kl)
                break

        with torch.no_grad():
            mean_after = actor.mean(obs).mean(dim=0)
            std_after = torch.exp(actor.clamped_log_std())

        metrics = {
            "actor_loss": float(np.mean(actor_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "critic_loss_before": critic_loss_before,
            "mean_reward": float(rewards.mean()),
            "approx_kl": approx_kl,
            "clip_fraction": float(np.mean(clip_fracs)),
            "epochs_run": float(epochs_run),
            "logp_check_max_abs_diff": mismatch,
        }
        for i, name in enumerate(("tx", "ty", "log_scale", "theta", "normal_offset")):
            metrics[f"mean_{name}"] = float(mean_after[i])
            metrics[f"std_{name}"] = float(std_after[i])
        logger.info(
            "PPO update on %d tuples: mean_reward=%.4f actor_loss=%.4f critic_loss=%.4f kl=%.4f clip=%.2f",
            n, metrics["mean_reward"], metrics["actor_loss"], metrics["critic_loss"],
            approx_kl, metrics["clip_fraction"],
        )
        return actor, critic, metrics
