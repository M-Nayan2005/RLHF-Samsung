"""
Environment-driven configuration for the serving_ui service.

Two rules this module enforces, both from AGENTS.md / docs/reference/open-questions.md:

1. **Never invent an unset hyperparameter.** The E-DRDE weights (alpha, beta,
   w1..w3) and the dual-metric router thresholds are deliberately unset in the
   spec. None of them belong to this service, and none are read here. If one
   ever shows up in this file, it is a bug.

2. **Every knob is named and defaulted in one place.** `WIGGLE_SIGMA` is the
   only tuning value here that comes from the shared `.env.example`; everything
   else is an engineering setting local to this track. Deviations from the spec
   are recorded in `docs/dev3-decisions.md`, not buried in a default.

Plain `os.environ` parsing rather than pydantic-settings, to avoid adding a
dependency the other three tracks do not already have.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


def _default_assets_dir() -> Path:
    """
    Locate `tests/assets/`, which sits at a different depth in each layout:

        laptop    <repo>/services/serving_ui/app/config.py  ->  <repo>/tests/assets
        container /app/app/config.py                        ->  /app/tests/assets

    Walking up until the directory is found handles both. A fixed `parents[N]`
    does not: `parents[3]` is correct on a laptop and raises IndexError in the
    container, where there are only three levels above the file. That crash took
    the service down at import time on the first real `docker compose up`.

    Never raises. A missing assets directory is not worth refusing to boot over -
    it only means `/assets` serves nothing, and this default is discarded anyway
    whenever ASSETS_DIR is set (as it is in .env.example).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "tests" / "assets"
        if candidate.is_dir():
            return candidate
    return here.parents[min(1, len(here.parents) - 1)] / "tests" / "assets"

# --------------------------------------------------------------------------
# Enumerated choices
# --------------------------------------------------------------------------

WIGGLE_MODES = ("affine", "vertex")
SCALE_REFERENCES = ("bbox_diagonal", "mask_diagonal")
TELEMETRY_TRANSPORTS = ("beacon", "ls_meta", "both")
STORE_BACKENDS = ("jsonl", "memory")
IMAGE_DIM_SOURCES = ("probe", "fixed")


def _get(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name, "true" if default else "false").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean, got {raw!r}")


def _get_float(name: str, default: float) -> float:
    raw = _get(name, str(default))
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc


def _get_int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an int, got {raw!r}") from exc


def _get_choice(name: str, default: str, allowed: tuple) -> str:
    value = _get(name, default).lower()
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    return value


@dataclass(frozen=True)
class Settings:
    # -- service identity -------------------------------------------------
    port: int
    log_level: str
    environment: str
    model_version: str

    # -- upstream (Dev 2) and downstream (Dev 4) --------------------------
    routing_qa_url: str
    webhook_gateway_url: str
    mock_mode: bool
    mock_task_path: str
    request_timeout_s: float

    # -- auth stub --------------------------------------------------------
    annotator_id_header: str
    default_annotator_id: str
    junior_queue_name: str

    # -- stochastic policy ------------------------------------------------
    wiggle_sigma: float
    wiggle_mode: str
    wiggle_scale_reference: str
    wiggle_clamp_to_frame: bool
    wiggle_seed_bytes: int

    # -- image geometry ---------------------------------------------------
    image_dim_source: str
    default_image_width: int
    default_image_height: int
    image_probe_timeout_s: float

    # -- locally served assets --------------------------------------------
    assets_dir: str
    assets_url_prefix: str

    # -- Label Studio wiring ----------------------------------------------
    label_studio_url: str
    label_studio_api_token: str
    ls_from_name: str
    ls_to_name: str
    ls_label_fallback: str
    ls_project_title: str

    # -- served-wiggle store (the Q11 / D10 mitigation) -------------------
    served_store_backend: str
    served_store_path: str

    # -- telemetry transport ----------------------------------------------
    telemetry_transport: str
    telemetry_forward_enabled: bool

    @property
    def prediction_model_version(self) -> str:
        """What Label Studio records as `prediction.model_version`."""
        return self.model_version

    def redacted(self) -> dict:
        """Config snapshot safe to log or return from /health."""
        data = {
            k: v for k, v in self.__dict__.items()
            if k not in ("label_studio_api_token",)
        }
        data["label_studio_api_token"] = (
            "<set>" if self.label_studio_api_token else "<unset>"
        )
        return data


def load_settings() -> Settings:
    """
    Read settings from the process environment.

    Raises ValueError on a malformed value rather than falling back to a
    default. A silently-wrong WIGGLE_SIGMA would poison every rollout in the
    batch and there would be nothing in the logs to explain it.
    """
    settings = Settings(
        port=_get_int("SERVING_UI_PORT", 8003),
        log_level=_get("LOG_LEVEL", "INFO").upper(),
        environment=_get("ENVIRONMENT", "development"),
        model_version=_get("SERVING_UI_MODEL_VERSION", "serving-ui-stochastic-0.1.0"),

        routing_qa_url=_get("ROUTING_QA_URL", "http://routing_qa:8002").rstrip("/"),
        webhook_gateway_url=_get("WEBHOOK_GATEWAY_URL", "http://webhook_gateway:8004").rstrip("/"),
        mock_mode=_get_bool("MOCK_MODE", False),
        mock_task_path=_get("MOCK_TASK_PATH", "/app/tests/mocks/routing_task.json"),
        request_timeout_s=_get_float("REQUEST_TIMEOUT_S", 5.0),

        annotator_id_header=_get("ANNOTATOR_ID_HEADER", "X-Annotator-Id"),
        default_annotator_id=_get("DEFAULT_ANNOTATOR_ID", "annotator_unknown"),
        junior_queue_name=_get("JUNIOR_QUEUE_NAME", "junior"),

        wiggle_sigma=_get_float("WIGGLE_SIGMA", 0.02),
        wiggle_mode=_get_choice("WIGGLE_MODE", "affine", WIGGLE_MODES),
        wiggle_scale_reference=_get_choice(
            "WIGGLE_SCALE_REFERENCE", "bbox_diagonal", SCALE_REFERENCES
        ),
        wiggle_clamp_to_frame=_get_bool("WIGGLE_CLAMP_TO_FRAME", True),
        wiggle_seed_bytes=_get_int("WIGGLE_SEED_BYTES", 16),

        image_dim_source=_get_choice("IMAGE_DIM_SOURCE", "probe", IMAGE_DIM_SOURCES),
        default_image_width=_get_int("DEFAULT_IMAGE_WIDTH", 1920),
        default_image_height=_get_int("DEFAULT_IMAGE_HEIGHT", 1080),
        image_probe_timeout_s=_get_float("IMAGE_PROBE_TIMEOUT_S", 3.0),

        # Images this service serves itself. Read from disk rather than fetched
        # over HTTP: a service probing its own listening socket is a needless
        # round trip that fails outright under a test client, where nothing is
        # listening.
        assets_dir=_get("ASSETS_DIR", str(_default_assets_dir())),
        assets_url_prefix=_get("ASSETS_URL_PREFIX", "/assets/"),

        label_studio_url=_get("LABEL_STUDIO_URL", "http://localhost:8080").rstrip("/"),
        label_studio_api_token=_get("LABEL_STUDIO_API_TOKEN", ""),
        ls_from_name=_get("LS_FROM_NAME", "polygon"),
        ls_to_name=_get("LS_TO_NAME", "image"),
        ls_label_fallback=_get("LS_LABEL_FALLBACK", "object"),
        ls_project_title=_get("LS_PROJECT_TITLE", "RLHF Segmentation - Junior Pool"),

        served_store_backend=_get_choice("SERVED_STORE_BACKEND", "jsonl", STORE_BACKENDS),
        served_store_path=_get("SERVED_STORE_PATH", "/app/data/served_wiggles.jsonl"),

        telemetry_transport=_get_choice("TELEMETRY_TRANSPORT", "beacon", TELEMETRY_TRANSPORTS),
        telemetry_forward_enabled=_get_bool("TELEMETRY_FORWARD_ENABLED", True),
    )

    _validate(settings)
    return settings


def _validate(s: Settings) -> None:
    if s.wiggle_sigma < 0:
        raise ValueError(f"WIGGLE_SIGMA must be >= 0, got {s.wiggle_sigma}")
    if s.wiggle_seed_bytes < 8:
        raise ValueError(
            f"WIGGLE_SEED_BYTES must be >= 8 so seeds stay collision-free, got {s.wiggle_seed_bytes}"
        )
    if s.default_image_width <= 0 or s.default_image_height <= 0:
        raise ValueError("DEFAULT_IMAGE_WIDTH and DEFAULT_IMAGE_HEIGHT must both be positive")


def startup_warnings(s: Settings) -> List[str]:
    """
    Non-fatal configuration smells worth shouting about at boot.

    Kept separate from `_validate` because none of these should stop the service
    from starting; they should stop a human from trusting the output.
    """
    warnings: List[str] = []

    if s.wiggle_sigma == 0:
        warnings.append(
            "WIGGLE_SIGMA=0 disables the stochastic policy entirely. The decoder "
            "becomes deterministic, every served mask equals the baseline, and the "
            "rollouts carry no exploration variance (Fault 3). Debug-only setting."
        )
    if s.mock_mode:
        warnings.append(
            f"MOCK_MODE=true: serving the fixture at {s.mock_task_path} instead of "
            f"calling routing_qa. Do not run an integration test in this mode."
        )
    if not s.label_studio_api_token:
        warnings.append(
            "LABEL_STUDIO_API_TOKEN is unset. /predict still works (Label Studio "
            "calls us), but label_studio/setup_project.py cannot configure the "
            "project or the webhook."
        )
    if s.image_dim_source == "fixed":
        warnings.append(
            f"IMAGE_DIM_SOURCE=fixed: every polygon will be converted to Label Studio "
            f"percentages against {s.default_image_width}x{s.default_image_height}. "
            f"Any image with different dimensions will render its mask in the wrong place."
        )
    if s.telemetry_transport == "ls_meta":
        warnings.append(
            "TELEMETRY_TRANSPORT=ls_meta relies on Label Studio Frontend meta "
            "injection, which stock Label Studio does not officially support. "
            "See services/serving_ui/README.md -> Telemetry transport."
        )
    return warnings


_SETTINGS: Optional[Settings] = None


def get_settings() -> Settings:
    """Process-wide singleton. Call `reset_settings()` in tests after monkeypatching env."""
    global _SETTINGS
    if _SETTINGS is None:
        _SETTINGS = load_settings()
    return _SETTINGS


def reset_settings() -> None:
    global _SETTINGS
    _SETTINGS = None
