"""
Shared test fixtures.

Also fixes `sys.path` so the suite runs from the repo root without installation.
Inside the container `/app` is already the working directory and both `app` and
`common` resolve; on a developer laptop the layout is
`services/serving_ui/app` + `common/`, so both roots go on the path here.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICE_ROOT = REPO_ROOT / "services" / "serving_ui"

for entry in (REPO_ROOT, SERVICE_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MOCKS = REPO_ROOT / "tests" / "mocks"


@pytest.fixture
def routing_task_raw() -> dict:
    return json.loads((MOCKS / "routing_task.json").read_text(encoding="utf-8"))


@pytest.fixture
def routing_task(routing_task_raw):
    from common.schemas.routing_queue import QueueTask

    return QueueTask.model_validate(routing_task_raw)


@pytest.fixture
def base_env(tmp_path, monkeypatch) -> dict:
    """
    A complete, self-contained environment for the service.

    Everything that would reach the network is disabled: MOCK_MODE replaces
    routing_qa, IMAGE_DIM_SOURCE=fixed replaces the image probe, and forwarding
    to the gateway is off. A test that needs one of those turns it back on
    explicitly, so no test can accidentally depend on a teammate's service.
    """
    env = {
        "MOCK_MODE": "true",
        "MOCK_TASK_PATH": str(MOCKS / "routing_task.json"),
        "IMAGE_DIM_SOURCE": "fixed",
        "DEFAULT_IMAGE_WIDTH": "800",
        "DEFAULT_IMAGE_HEIGHT": "600",
        "SERVED_STORE_PATH": str(tmp_path / "served.jsonl"),
        "TELEMETRY_FORWARD_ENABLED": "false",
        "WIGGLE_SIGMA": "0.02",
        "LOG_LEVEL": "WARNING",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


@pytest.fixture
def settings(base_env):
    from app.config import load_settings

    return load_settings()


@pytest.fixture
def client(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def label_config() -> str:
    return (SERVICE_ROOT / "label_studio" / "labeling_config.xml").read_text(encoding="utf-8")
