"""
The trainable wiggle policy, shared by Tier 2 (serving) and Tier 4 (training).

Tier 2's stochastic decoder (`services/serving_ui/app/wiggle.py`) perturbs the
Tier 1 consensus mask with a 5-dimensional affine action
`(tx, ty, log_scale, theta, normal_offset)`. Until now that action was drawn
from a fixed `N(0, I)`: exploration with nothing to learn. This module makes it
a policy:

    z ~ pi_theta(. | s) = N(mu_theta(s), diag(sigma_theta^2))

where `s` is a small observation vector computed from the state the decoder
acts on (the consensus mask, the image frame, and Tier 1's uncertainty
metrics). `wiggle.py` keeps applying `z` exactly as before - `z` is the
standardised draw it already scaled by `WIGGLE_SIGMA` - so a policy with
`mu = 0, sigma = 1` reproduces the old behaviour bit for bit from the same seed.

Why a small Gaussian policy instead of LoRA on the SAM2 decoder: see
`docs/tier4-policy.md`. In short, the action Tier 2 actually emits is these
five numbers, so this is the thing PPO can attribute reward to, it trains on a
CPU, and it closes the loop today. SAM2 fine-tuning can be layered on later
without changing the contracts.

This file is pure numpy so the serving container does not need torch. Tier 4
re-implements `mean` / `std` / `log_prob` in torch and a test pins the two
implementations to each other (`services/tier4_worker/tests/test_parity.py`).
Change one, change both.

Checkpoint layout under `<models_dir>/policy/`:

    current.json            {"version": "..."} - the version being served
    <version>.json          the policy weights (this module's format)
    <version>.critic.pt     Tier 4's value network, torch-only, optional
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Bump if the observation features change. Stored on every checkpoint so a
# policy is never evaluated against features it was not trained on.
OBS_SPEC = "rl-obs-1"
OBS_FEATURES = (
    "log_area_fraction",      # log(area / image area) / 5
    "log_aspect",             # log(bbox width / bbox height)
    "centroid_x",             # centroid x / W - 0.5
    "centroid_y",             # centroid y / H - 0.5
    "compactness",            # 4 pi A / P^2, 1 for a circle
    "log_vertices",           # log(vertex count) / 5
    "geometric_variance",     # Tier 1 MCD spread
    "class_logit_entropy",    # Tier 1 class uncertainty
)
OBS_DIM = len(OBS_FEATURES)
ACTION_DIM = 5
HIDDEN_DIM = 32

# Guardrails, applied identically in numpy and torch so log-probs match.
# MEAN_LIMIT bounds how far the policy can push the mean wiggle (in units of
# WIGGLE_SIGMA * length scale); the log-std bounds keep exploration alive
# without letting the policy blow the mask apart.
MEAN_LIMIT = 3.0
LOG_STD_MIN = -2.0
LOG_STD_MAX = 0.5
_OBS_CLIP = 5.0
_LOG_2PI = math.log(2.0 * math.pi)


class PolicyError(ValueError):
    """A checkpoint is missing, malformed, or built for different features."""


# --------------------------------------------------------------------------
# Observation
# --------------------------------------------------------------------------

def observation(
    points: Sequence[Sequence[float]],
    image_width: float,
    image_height: float,
    geometric_variance: float = 0.0,
    class_logit_entropy: float = 0.0,
) -> List[float]:
    """The state vector `s` the policy conditions on. Order is `OBS_FEATURES`."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        raise PolicyError("observation needs a polygon of at least 3 [x, y] points")
    if image_width <= 0 or image_height <= 0:
        raise PolicyError("observation needs positive image dimensions")

    x, y = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area = abs(cross.sum()) / 2.0
    perimeter = float(np.hypot(x2 - x, y2 - y).sum())

    if area > 1e-9:
        cx = float(((x + x2) * cross).sum() / (6.0 * (cross.sum() / 2.0)))
        cy = float(((y + y2) * cross).sum() / (6.0 * (cross.sum() / 2.0)))
    else:
        cx, cy = float(x.mean()), float(y.mean())

    bw = max(float(x.max() - x.min()), 1e-6)
    bh = max(float(y.max() - y.min()), 1e-6)
    frame = float(image_width) * float(image_height)

    feats = [
        math.log(max(area, 1e-6) / frame) / 5.0,
        math.log(bw / bh),
        cx / float(image_width) - 0.5,
        cy / float(image_height) - 0.5,
        (4.0 * math.pi * area / (perimeter ** 2)) if perimeter > 1e-9 else 0.0,
        math.log(pts.shape[0]) / 5.0,
        float(geometric_variance),
        float(class_logit_entropy),
    ]
    return [float(np.clip(f, -_OBS_CLIP, _OBS_CLIP)) for f in feats]


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass
class PolicyParams:
    """
    A one-hidden-layer Gaussian policy.

        h    = tanh(W1 s + b1)
        mu   = MEAN_LIMIT * tanh((W2 h + b2) / MEAN_LIMIT)
        std  = exp(clip(log_std, LOG_STD_MIN, LOG_STD_MAX))      state-independent

    `W2 = 0, b2 = 0, log_std = 0` gives `mu = 0, std = 1`: the original fixed
    N(0, I) wiggle. That is the initial checkpoint, so turning learning on does
    not change what annotators see until a batch has actually been trained.
    """
    version: str
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    log_std: np.ndarray
    parent_version: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)
    obs_spec: str = OBS_SPEC

    # -- evaluation -------------------------------------------------------

    def _check(self, obs: Sequence[float]) -> np.ndarray:
        s = np.asarray(obs, dtype=np.float64)
        if s.shape != (OBS_DIM,):
            raise PolicyError(f"observation must have {OBS_DIM} features, got shape {s.shape}")
        return s

    def mean(self, obs: Sequence[float]) -> np.ndarray:
        s = self._check(obs)
        h = np.tanh(self.w1 @ s + self.b1)
        return MEAN_LIMIT * np.tanh((self.w2 @ h + self.b2) / MEAN_LIMIT)

    def std(self) -> np.ndarray:
        return np.exp(np.clip(self.log_std, LOG_STD_MIN, LOG_STD_MAX))

    def log_prob(self, obs: Sequence[float], z: Sequence[float]) -> float:
        mu = self.mean(obs)
        std = self.std()
        a = np.asarray(z, dtype=np.float64)
        return float(np.sum(-0.5 * ((a - mu) / std) ** 2 - np.log(std) - 0.5 * _LOG_2PI))

    def is_identity(self) -> bool:
        """True when this policy is exactly the old fixed N(0, I) wiggle."""
        return (not np.any(self.w2)) and (not np.any(self.b2)) and (not np.any(self.log_std))

    # -- serialisation ----------------------------------------------------

    def to_dict(self) -> Dict:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "created_at": self.created_at,
            "obs_spec": self.obs_spec,
            "obs_dim": OBS_DIM,
            "hidden_dim": int(self.w1.shape[0]),
            "action_dim": ACTION_DIM,
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2.tolist(),
            "log_std": self.log_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "PolicyParams":
        if data.get("obs_spec") != OBS_SPEC:
            raise PolicyError(
                f"checkpoint {data.get('version')!r} was built for observation spec "
                f"{data.get('obs_spec')!r}, this code computes {OBS_SPEC!r}"
            )
        try:
            params = cls(
                version=str(data["version"]),
                w1=np.asarray(data["w1"], dtype=np.float64),
                b1=np.asarray(data["b1"], dtype=np.float64),
                w2=np.asarray(data["w2"], dtype=np.float64),
                b2=np.asarray(data["b2"], dtype=np.float64),
                log_std=np.asarray(data["log_std"], dtype=np.float64),
                parent_version=data.get("parent_version"),
                created_at=data.get("created_at") or utc_now_iso(),
            )
        except KeyError as exc:
            raise PolicyError(f"checkpoint is missing {exc}") from exc
        hidden = params.w1.shape[0] if params.w1.ndim == 2 else -1
        if (
            params.w1.shape != (hidden, OBS_DIM) or params.b1.shape != (hidden,)
            or params.w2.shape != (ACTION_DIM, hidden) or params.b2.shape != (ACTION_DIM,)
            or params.log_std.shape != (ACTION_DIM,)
        ):
            raise PolicyError(f"checkpoint {params.version!r} has inconsistent weight shapes")
        return params


def initial_policy(version: str, hidden_dim: int = HIDDEN_DIM) -> PolicyParams:
    """
    The bootstrap checkpoint: mean 0, std 1, i.e. the original wiggle.

    `W1` is random but seeded, so two services bootstrapping at the same moment
    write byte-identical files. It must be non-zero or the hidden layer starts
    dead and `W2`'s gradient has nothing to work with.
    """
    rng = np.random.default_rng(0)
    return PolicyParams(
        version=version,
        w1=rng.standard_normal((hidden_dim, OBS_DIM)) / math.sqrt(OBS_DIM),
        b1=np.zeros(hidden_dim),
        w2=np.zeros((ACTION_DIM, hidden_dim)),
        b2=np.zeros(ACTION_DIM),
        log_std=np.zeros(ACTION_DIM),
    )


def next_version(version: str) -> str:
    """`serving-ui-stochastic-0.1.0` -> `serving-ui-stochastic-0.1.1`."""
    head, sep, tail = version.rpartition(".")
    if sep and tail.isdigit():
        return f"{head}.{int(tail) + 1}"
    return f"{version}.1"


# --------------------------------------------------------------------------
# Checkpoint store
# --------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class PolicyStore:
    """
    The `<models_dir>/policy/` directory, read by serving and written by Tier 4.

    `current()` is cheap enough to call on every request: it stats one small
    file and only re-reads weights when the pointer changes.
    """

    def __init__(self, models_dir: str, default_version: str) -> None:
        self.root = Path(models_dir) / "policy"
        self.default_version = default_version
        self._cached: Optional[PolicyParams] = None
        self._cached_stamp: Optional[tuple] = None

    @property
    def pointer(self) -> Path:
        return self.root / "current.json"

    def path_for(self, version: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in version)
        return self.root / f"{safe}.json"

    def load(self, version: str) -> PolicyParams:
        path = self.path_for(version)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PolicyError(f"no checkpoint for version {version!r} at {path}") from exc
        except json.JSONDecodeError as exc:
            raise PolicyError(f"checkpoint {path} is not valid JSON: {exc}") from exc
        return PolicyParams.from_dict(data)

    def save(self, params: PolicyParams, make_current: bool = True) -> Path:
        """Write a checkpoint, then (optionally) flip the pointer to it. Both writes are atomic."""
        path = self.path_for(params.version)
        _atomic_write_json(path, params.to_dict())
        if make_current:
            _atomic_write_json(self.pointer, {"version": params.version, "updated_at": utc_now_iso()})
        return path

    def bootstrap(self) -> PolicyParams:
        """Create the initial identity checkpoint if no policy has ever been published."""
        if not self.pointer.exists():
            params = initial_policy(self.default_version)
            self.save(params, make_current=True)
        return self.current()

    def current(self) -> PolicyParams:
        try:
            st = self.pointer.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except FileNotFoundError:
            return self.bootstrap()

        if self._cached is not None and stamp == self._cached_stamp:
            return self._cached

        data = json.loads(self.pointer.read_text(encoding="utf-8"))
        params = self.load(str(data["version"]))
        self._cached, self._cached_stamp = params, stamp
        return params
