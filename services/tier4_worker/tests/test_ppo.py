"""
Tier 4 PPO: the torch policy matches the numpy one serving uses, and PPO moves
the policy toward reward on a bandit with a known optimum.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
for entry in (REPO_ROOT, REPO_ROOT / "services"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from common import rl_policy  # noqa: E402
from tier4_worker.deploy import PolicyPublisher  # noqa: E402
from tier4_worker.ppo import OffPolicyBatchError, PPOConfig, PPOTrainer, TorchPolicy  # noqa: E402

RNG = np.random.default_rng(1)


def _random_params(version="v-test-0.0.0"):
    p = rl_policy.initial_policy(version)
    p.w2 = RNG.standard_normal(p.w2.shape) * 0.3
    p.b2 = RNG.standard_normal(5) * 0.2
    p.log_std = RNG.uniform(-0.5, 0.3, 5)
    return p


def _obs():
    pts = [[50 + 30 * np.cos(a), 60 + 20 * np.sin(a)] for a in np.linspace(0, 2 * np.pi, 9)[:-1]]
    return rl_policy.observation(pts, 640, 480, RNG.uniform(0, 0.05), RNG.uniform(0, 0.2))


def test_torch_policy_matches_numpy_policy():
    params = _random_params()
    torch_policy = TorchPolicy(params)
    for _ in range(20):
        obs = _obs()
        z = RNG.standard_normal(5) * 2
        expected = params.log_prob(obs, z)
        got = float(torch_policy.log_prob(torch.tensor([obs], dtype=torch.float64),
                                          torch.tensor([z], dtype=torch.float64))[0])
        assert got == pytest.approx(expected, abs=1e-9)
        assert np.allclose(torch_policy.mean(torch.tensor(obs, dtype=torch.float64)).detach().numpy(),
                           params.mean(obs), atol=1e-12)


def test_export_round_trips_through_the_checkpoint_format():
    params = _random_params()
    exported = TorchPolicy(params).export("v-test-0.0.1", parent_version=params.version)
    again = rl_policy.PolicyParams.from_dict(exported.to_dict())
    obs, z = _obs(), RNG.standard_normal(5)
    assert again.log_prob(obs, z) == pytest.approx(params.log_prob(obs, z), abs=1e-12)
    assert again.parent_version == params.version


def _rollouts(policy, n, optimum):
    rows = []
    for _ in range(n):
        obs = _obs()
        z = policy.mean(obs) + policy.std() * RNG.standard_normal(5)
        reward = -float(np.sum((z - optimum) ** 2))
        rows.append({"policy_obs": obs, "action_z": z.tolist(),
                     "action_log_prob": policy.log_prob(obs, z), "reward_r_t": reward})
    return rows


def test_ppo_improves_reward_on_a_known_bandit():
    """Reward peaks at z = optimum. A few on-policy batches must move the mean there."""
    optimum = np.array([0.0, 0.0, 0.0, 0.0, 1.5])
    policy = rl_policy.initial_policy("bandit-0.0.0")
    trainer = PPOTrainer(PPOConfig(actor_lr=3e-2, target_kl=1.0), seed=0)
    critic_state = None

    def expected_reward(p):
        return float(np.mean([-np.sum((p.mean(_obs()) - optimum) ** 2) - np.sum(p.std() ** 2) for _ in range(50)]))

    before = expected_reward(policy)
    for _ in range(15):
        actor, critic, metrics = trainer.train(policy, _rollouts(policy, 64, optimum), critic_state)
        policy = actor.export(rl_policy.next_version(policy.version), policy.version)
        critic_state = critic.state_dict()
    after = expected_reward(policy)

    assert after > before + 1.0, (before, after)
    assert np.mean([policy.mean(_obs())[4] for _ in range(20)]) > 0.5
    assert policy.version == "bandit-0.0.15"


def test_a_batch_from_another_checkpoint_is_refused():
    served_by = _random_params("v-a")
    rows = _rollouts(served_by, 64, np.zeros(5))
    with pytest.raises(OffPolicyBatchError):
        PPOTrainer().train(rl_policy.initial_policy("v-b"), rows)


def test_publish_flips_the_pointer_and_keeps_the_old_checkpoint(tmp_path):
    publisher = PolicyPublisher(str(tmp_path), "pub-0.1.0")
    first = publisher.store.bootstrap()
    actor, critic, _ = PPOTrainer(seed=0).train(first, _rollouts(first, 64, np.ones(5)))
    new = actor.export(rl_policy.next_version(first.version), first.version)
    publisher.publish(new, critic.state_dict())

    assert publisher.store.current().version == "pub-0.1.1"
    assert publisher.store.load("pub-0.1.0").is_identity()
    assert publisher.load_critic("pub-0.1.1") is not None
