"""
The stochastic mask decoder - Tier 2's RL policy head.

Tier 2 spec, section 2: the SAM2 mask decoder must not behave deterministically.
It samples its output from a Gaussian `N(mu, sigma^2)`, where `mu` is the
model's optimal guess (the `baseline_mask` handed over by Tier 1 via Dev 2's
`QueueTask`) and `sigma^2` is noise injected on purpose. The resulting "wiggled"
mask is deliberately slightly wrong. That is the exploration mechanism: a
deterministic decoder emits the same mask every time, generates no variance, and
an RL agent with no variance cannot discover what humans prefer (**Fault 3**).

Two things this module is careful about.

**Low-dimensional action space (Fault 2).** The spec is explicit that the action
space is "low-dimensional latent points and boxes, never a raw pixel grid",
because the critic has to attribute reward to a handful of parameters rather
than to a million pixels. Perturbing every vertex independently would make the
action 2N-dimensional - 400 dimensions for a 200-vertex polygon - which is the
credit-assignment problem the spec is explicitly avoiding. So the default mode
here is `affine`: a **5-dimensional** action vector, shared across the whole
polygon. `vertex` mode is implemented too, because the Dev 3 brief asks for a
documented choice between the two, but it is not the default and the reason is
written down in `docs/dev3-decisions.md` (DD-2).

**Bit-reproducibility.** `LSTelemetryMeta.wiggle_seed` is the only record of the
wiggled polygon that the frozen contracts carry, so Tier 3 can only recover the
sampled action `A_t` if the perturbation is exactly reproducible from that seed.
See "Reproducibility contract" below and open question Q11 / divergence D10.
This service also persists the served polygon outright (`store.py`), because a
reconstruction path that depends on three invariants holding forever is not a
sound foundation for a reward signal.

Reproducibility contract
------------------------
Given the same `(baseline_points, seed, sigma, mode, scale_reference,
length_scale)`, `apply_wiggle` returns bit-identical output. That holds because:

* **RNG is pinned** to numpy's PCG64 via `np.random.Generator`. numpy guarantees
  stream compatibility for `Generator` bit generators across versions; the
  legacy `RandomState` and Python's `random` module make weaker promises.
* **Seed derivation is pinned** to BLAKE2b-128 of the UTF-8 seed string, read
  big-endian. Any string works as a seed, including seeds minted elsewhere.
* **Draw order is pinned** and documented per mode below. Adding a draw in the
  middle silently invalidates every seed ever issued - append only.
* **Vertex order is pinned** by `geometry.canonicalize`, so an upstream service
  that flips its winding or starts the ring at a different vertex still yields
  the same wiggle.

Any change to the above is a breaking change to already-issued seeds. Bump
`WIGGLE_ALGORITHM_VERSION` and record it in `docs/dev3-decisions.md`.
"""
from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import geometry

# Bump on ANY change to draw order, seed derivation, or the transform itself.
# Persisted alongside every served wiggle so an old seed can still be replayed
# with the algorithm that actually produced it.
WIGGLE_ALGORITHM_VERSION = "wiggle-1.0.0"

# Fixed draw order. Append-only: inserting a name shifts every later draw.
AFFINE_ACTION_DIMS = ("tx", "ty", "log_scale", "theta", "normal_offset")


class WiggleError(ValueError):
    """Raised when a polygon cannot be wiggled (degenerate input, bad config)."""


# --------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------

def new_seed(n_bytes: int = 16) -> str:
    """
    Mint a fresh seed for one served task.

    Random rather than derived from `task_id`, so that a task re-served after a
    consensus requeue gets genuinely fresh exploration noise instead of
    replaying the same action the annotator already rejected.
    """
    if n_bytes < 8:
        raise WiggleError(f"seed needs >= 8 bytes to stay collision-free, got {n_bytes}")
    return secrets.token_hex(n_bytes)


def seed_to_int(seed: str) -> int:
    """
    Map an arbitrary seed string to a 128-bit integer for PCG64.

    BLAKE2b rather than `hash()` (randomised per process by PYTHONHASHSEED) or
    `int(seed, 16)` (would reject any non-hex seed, including the `seed_7781`
    style used in the shared fixtures).
    """
    digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=16).digest()
    return int.from_bytes(digest, byteorder="big")


def rng_from_seed(seed: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed_to_int(seed)))


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class WiggleParams:
    """
    Everything needed to replay a wiggle, minus the baseline polygon itself.

    Persisted with every served task. Tier 3 needs all of it: replaying with the
    right seed but a stale sigma reproduces a different action.
    """
    seed: str
    sigma: float
    mode: str
    scale_reference: str
    length_scale: float
    clamped: bool
    algorithm_version: str = WIGGLE_ALGORITHM_VERSION

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass(frozen=True)
class WiggleResult:
    points: List[List[float]]
    baseline_points: List[List[float]]
    params: WiggleParams
    action: Dict[str, float]
    diagnostics: Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "points": self.points,
            "baseline_points": self.baseline_points,
            "params": self.params.to_dict(),
            "action": self.action,
            "diagnostics": self.diagnostics,
        }


# --------------------------------------------------------------------------
# Length scale
# --------------------------------------------------------------------------

def resolve_length_scale(
    canonical: np.ndarray,
    scale_reference: str,
    bounding_box: Optional[Sequence[float]] = None,
) -> float:
    """
    Pick the pixel length that `sigma` is measured against.

    `sigma` is dimensionless in `.env.example` (`WIGGLE_SIGMA=0.02`) and the spec
    never says what it scales. Multiplying raw pixel coordinates by 0.02 would
    make the perturbation depend on how far the object happens to sit from the
    image origin, which is meaningless. Scaling by a length intrinsic to the
    object makes sigma a *relative* displacement: sigma=0.02 means "about 2% of
    the object's diagonal". Recorded as divergence D14 / question Q14 - the
    interpretation is this track's choice, not the spec's.

    `bbox_diagonal` (default) uses Grounding DINO's box from the `QueueTask`,
    which is the spec's own "points and boxes" action-space element and is
    stable even when the mask is ragged. Falls back to the mask's own extent
    when the box is missing or degenerate.
    """
    if scale_reference == "bbox_diagonal" and bounding_box is not None:
        x_min, y_min, x_max, y_max = (float(v) for v in bounding_box)
        length = geometry.diagonal(x_min, y_min, x_max, y_max)
        if length > 1e-6:
            return length

    length = geometry.diagonal(*geometry.bbox(canonical))
    if length <= 1e-6:
        raise WiggleError("polygon has zero extent; cannot derive a length scale")
    return length


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------

def apply_wiggle(
    baseline_points: Sequence[Sequence[float]],
    *,
    seed: str,
    sigma: float,
    mode: str = "affine",
    scale_reference: str = "bbox_diagonal",
    bounding_box: Optional[Sequence[float]] = None,
    image_width: Optional[float] = None,
    image_height: Optional[float] = None,
    clamp: bool = True,
) -> WiggleResult:
    """
    Sample one action from the policy and apply it to the baseline mask.

    `baseline_points` is `QueueTask.baseline_mask.points` - absolute pixels, the
    distribution mean `mu`. The returned polygon is the sampled action `A_t`.
    """
    if sigma < 0:
        raise WiggleError(f"sigma must be >= 0, got {sigma}")

    try:
        arr = geometry.as_array(baseline_points)
    except ValueError as exc:
        raise WiggleError(str(exc)) from exc

    canonical = geometry.canonicalize(arr)

    # The mask's own extent is checked before the length scale is resolved,
    # because `bbox_diagonal` takes its scale from Grounding DINO's box - so a
    # collapsed or collinear mask paired with a healthy box would sail through
    # and be served as a zero-area polygon. Label Studio renders that as an
    # invisible region, the annotator draws from scratch, and the telemetry
    # describes a correction to something that was never shown.
    if geometry.diagonal(*geometry.bbox(canonical)) <= 1e-6:
        raise WiggleError("baseline mask has zero extent; every vertex is the same point")
    if geometry.area(canonical) <= 0.0:
        raise WiggleError("baseline mask has zero area; its vertices are collinear")

    length_scale = resolve_length_scale(canonical, scale_reference, bounding_box)
    rng = rng_from_seed(seed)

    if mode == "affine":
        wiggled, action = _affine_wiggle(canonical, rng, sigma, length_scale)
    elif mode == "vertex":
        wiggled, action = _vertex_wiggle(canonical, rng, sigma, length_scale)
    else:
        raise WiggleError(f"unknown wiggle mode {mode!r}")

    clamped = False
    if clamp and image_width and image_height:
        before = wiggled
        wiggled = geometry.clamp_to_frame(wiggled, image_width, image_height)
        clamped = not np.allclose(before, wiggled)

    params = WiggleParams(
        seed=seed,
        sigma=sigma,
        mode=mode,
        scale_reference=scale_reference,
        length_scale=length_scale,
        clamped=clamped,
    )

    return WiggleResult(
        points=geometry.as_points(wiggled),
        baseline_points=geometry.as_points(canonical),
        params=params,
        action=action,
        diagnostics=_diagnostics(canonical, wiggled, length_scale),
    )


def _affine_wiggle(canonical, rng, sigma, length_scale):
    """
    5-dimensional action, shared across the whole polygon.

    Draw order (PINNED - append only):
        0 tx             translation along x
        1 ty             translation along y
        2 log_scale      isotropic dilation, in log space
        3 theta          rotation about the area centroid, radians
        4 normal_offset  uniform push along the outward boundary normal

    All five are drawn in a single `standard_normal(5)` call so the draw order is
    a property of the array, not of the statement order below.

    Scale is exponentiated (`exp(sigma * z)`) rather than added, so the
    perturbation is symmetric in log space: a shrink and the matching growth are
    equally likely, and the polygon can never invert through zero.

    `normal_offset` is what makes this more than a rigid-body move. Translation,
    rotation and scale together can only relocate a shape; the normal offset
    dilates the boundary itself, which is the error mode annotators actually
    correct - a mask that is slightly too fat or too thin along its edge. It is
    also the dimension that corresponds most directly to Tier 1's Deliberate
    Degradation Engine (a ~20 px polygon expansion), which is why it is here.
    See DD-3 in `docs/dev3-decisions.md`.
    """
    z = rng.standard_normal(5)
    tx, ty, log_scale, theta, normal_offset = (float(v) for v in z)

    dx = tx * sigma * length_scale
    dy = ty * sigma * length_scale
    scale = float(np.exp(log_scale * sigma))
    angle = theta * sigma
    offset = normal_offset * sigma * length_scale

    center = geometry.centroid(canonical)
    normals = geometry.outward_normals(canonical)

    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64)

    out = (canonical - center) @ rotation.T * scale + center
    out = out + normals * offset
    out = out + np.array([dx, dy], dtype=np.float64)

    action = {
        "tx_px": dx,
        "ty_px": dy,
        "scale": scale,
        "theta_rad": angle,
        "normal_offset_px": offset,
    }
    return out, action


def _vertex_wiggle(canonical, rng, sigma, length_scale):
    """
    2N-dimensional action: every vertex perturbed independently.

    Draw order (PINNED): a single `standard_normal((N, 2))` in canonical vertex
    order, x before y.

    Not the default. It produces a visibly noisier, more "jagged" mask that
    reads as a broken model rather than a near-miss, and it hands the critic a
    2N-dimensional credit-assignment problem - exactly what Fault 2 warns
    against. Kept for A/B comparison and because the Dev 3 brief asks for both
    options to be considered explicitly.
    """
    n = canonical.shape[0]
    z = rng.standard_normal((n, 2))
    delta = z * sigma * length_scale
    out = canonical + delta

    magnitudes = np.linalg.norm(delta, axis=1)
    action = {
        "n_vertices": float(n),
        "mean_vertex_shift_px": float(magnitudes.mean()),
        "max_vertex_shift_px": float(magnitudes.max()),
    }
    return out, action


def _diagnostics(canonical: np.ndarray, wiggled: np.ndarray, length_scale: float) -> Dict[str, float]:
    """
    Numbers a human can eyeball to tell whether the wiggle is doing its job.

    `iou_vs_baseline` near 1.0 means the wiggle is invisible and the annotator
    has nothing to correct (no exploration signal); near 0 means the mask is so
    wrong the annotator will redraw from scratch (no *gradient* signal either,
    since the correction no longer relates to the action). Surfaced on
    /wiggle/preview so sigma can be calibrated by looking rather than guessing.
    """
    displacement = np.linalg.norm(wiggled - canonical, axis=1)
    baseline_area = geometry.area(canonical)
    wiggled_area = geometry.area(wiggled)
    return {
        "iou_vs_baseline": geometry.raster_iou(canonical, wiggled),
        "mean_displacement_px": float(displacement.mean()),
        "max_displacement_px": float(displacement.max()),
        "displacement_over_length_scale": float(displacement.mean() / max(length_scale, 1e-9)),
        "area_ratio": float(wiggled_area / baseline_area) if baseline_area > 1e-9 else 0.0,
    }


# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------

def replay(baseline_points: Sequence[Sequence[float]], params: WiggleParams, **kwargs) -> WiggleResult:
    """
    Reproduce a previously served wiggle from its persisted params.

    This is the call Tier 3 makes to recover `A_t` if it only has the seed. It
    refuses to guess across algorithm versions: a seed minted by wiggle-1.0.0
    means nothing to a future wiggle-2.0.0 transform, and silently returning the
    wrong polygon would corrupt every reward computed from it.
    """
    if params.algorithm_version != WIGGLE_ALGORITHM_VERSION:
        raise WiggleError(
            f"cannot replay a wiggle from {params.algorithm_version} with "
            f"{WIGGLE_ALGORITHM_VERSION}; fetch the served polygon from the store instead"
        )
    return apply_wiggle(
        baseline_points,
        seed=params.seed,
        sigma=params.sigma,
        mode=params.mode,
        scale_reference=params.scale_reference,
        **kwargs,
    )
