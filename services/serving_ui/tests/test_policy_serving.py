"""
The learned policy on the serve path, and the Tier 3 wiggle-cache bridge.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from common import rl_policy
from common.schemas.tier3_rlhf import WiggleCacheEntry


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttl = {}

    def set(self, key, value, ex=None):
        self.data[key] = value
        self.ttl[key] = ex
        return True

    def close(self):
        pass


SQUARE = [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]]


def test_identity_policy_is_bit_identical_to_the_fixed_wiggle():
    from app.wiggle import apply_wiggle

    plain = apply_wiggle(SQUARE, seed="seed_1", sigma=0.02)
    with_identity = apply_wiggle(
        SQUARE, seed="seed_1", sigma=0.02, policy_mean=[0.0] * 5, policy_std=[1.0] * 5
    )
    assert with_identity.points == plain.points
    assert with_identity.action_z == plain.action_z


def test_policy_mean_shifts_the_draw_and_replays():
    from app.wiggle import apply_wiggle, replay

    plain = apply_wiggle(SQUARE, seed="seed_1", sigma=0.02)
    shifted = apply_wiggle(
        SQUARE, seed="seed_1", sigma=0.02, policy_mean=[0, 0, 0, 0, 2.0], policy_std=[0.5] * 5
    )
    expected = np.array(plain.action_z) * 0.5 + np.array([0, 0, 0, 0, 2.0])
    assert np.allclose(shifted.action_z, expected)
    assert replay(SQUARE, shifted.params).points == shifted.points


def test_vertex_mode_refuses_a_policy():
    from app.wiggle import WiggleError, apply_wiggle

    with pytest.raises(WiggleError):
        apply_wiggle(SQUARE, seed="s", sigma=0.02, mode="vertex", policy_mean=[1, 0, 0, 0, 0])


def test_predict_records_the_policy_action_and_writes_the_cache(settings, label_config):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings)
    fake = FakeRedis()
    app.state.wiggle_cache._client = fake
    app.state.wiggle_cache.enabled = True

    with TestClient(app) as client:
        response = client.post("/predict", json={"tasks": [{"id": 1, "data": {}}], "label_config": label_config})
    assert response.status_code == 200
    body = response.json()
    assert body["results"][0]["result"], "expected a served region"

    [(key, raw)] = fake.data.items()
    entry = WiggleCacheEntry.model_validate_json(raw)
    assert key == f"wiggle_cache:{entry.wiggle_seed}"
    assert fake.ttl[key] == settings.wiggle_cache_ttl_s

    # The initial checkpoint is bootstrapped and named after SERVING_UI_MODEL_VERSION.
    assert entry.model_version == settings.model_version
    assert body["model_version"] == settings.model_version
    assert len(entry.policy_obs) == rl_policy.OBS_DIM
    assert len(entry.action_z) == rl_policy.ACTION_DIM

    policy = rl_policy.PolicyStore(settings.models_dir, settings.model_version).current()
    assert entry.action_log_prob == pytest.approx(policy.log_prob(entry.policy_obs, entry.action_z))

    # m_initial is Tier 1's mask exactly; m_wiggled is what was served.
    served = app.state.store.by_seed(entry.wiggle_seed)
    assert entry.m_wiggled.points == served.wiggled_points
    assert entry.image_width == served.image_width


def test_a_newly_published_policy_is_served_without_a_restart(settings, label_config):
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app(settings)
    fake = FakeRedis()
    app.state.wiggle_cache._client = fake
    app.state.wiggle_cache.enabled = True
    store = rl_policy.PolicyStore(settings.models_dir, settings.model_version)

    with TestClient(app) as client:
        client.post("/predict", json={"tasks": [{"id": 1, "data": {}}], "label_config": label_config})
        trained = store.current()
        trained.version = rl_policy.next_version(trained.version)
        trained.b2 = np.array([0.0, 0.0, 0.0, 0.0, 1.5])
        store.save(trained)
        client.post("/predict", json={"tasks": [{"id": 2, "data": {}}], "label_config": label_config})

    versions = sorted(json.loads(raw)["model_version"] for raw in fake.data.values())
    assert versions == [settings.model_version, rl_policy.next_version(settings.model_version)]


def test_policy_disabled_serves_without_a_policy_action(base_env, monkeypatch, label_config):
    from fastapi.testclient import TestClient

    from app.config import load_settings
    from app.main import create_app

    monkeypatch.setenv("POLICY_ENABLED", "false")
    app = create_app(load_settings())
    fake = FakeRedis()
    app.state.wiggle_cache._client = fake
    app.state.wiggle_cache.enabled = True
    with TestClient(app) as client:
        client.post("/predict", json={"tasks": [{"id": 1, "data": {}}], "label_config": label_config})

    entry = WiggleCacheEntry.model_validate_json(next(iter(fake.data.values())))
    assert entry.action_z is None and entry.action_log_prob is None
