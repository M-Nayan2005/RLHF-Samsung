"""
Turning a wiggled `QueueTask` into a Label Studio `predictions` block.

Three conversions happen here, and each one is a place the integration can break
silently rather than loudly, so each is spelled out.

**Pixels to percentages.** `PolygonMask.points` are absolute pixels; Label Studio
polygon regions are percentages of the image, 0-100. See `image_meta.py` for how
the dimensions are obtained (divergence D12).

**Tag names.** A region whose `from_name` does not match a control tag in the
project's labeling config is dropped by Label Studio without an error - the
annotator just sees an empty canvas and assumes the model returned nothing.
Rather than trusting `LS_FROM_NAME` to stay in sync with the XML by hand, this
module parses the `label_config` that Label Studio sends along with every
`/predict` call and reads the real tag names out of it.

**The seed round-trip.** `wiggle_seed` has to reach the browser so the telemetry
beacon can attach it, and it has to survive Label Studio's own serialisation.
Three independent channels carry it, because each can fail on its own:

  1. `region.meta.text` - Label Studio's documented per-region metadata channel,
     preserved through storage and exposed to the frontend.
  2. `model_version` - a plain string Label Studio always round-trips verbatim.
  3. `GET /served/{task_id}` on this service - authoritative, and the fallback
     the instrumentation script uses when neither of the above is readable.

Belt and braces, deliberately: losing the seed does not break the annotation, it
breaks the link between the effort telemetry and the action that caused it,
which is the entire reward signal.
"""
from __future__ import annotations

import logging
import re
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from common.schemas.routing_queue import QueueTask

from .config import Settings
from .image_meta import ImageDims
from .models import LSPredictionResult
from .wiggle import WiggleResult

log = logging.getLogger(__name__)

SEED_META_PREFIX = "wiggle_seed="
MODEL_VERSION_SEED_SEPARATOR = "|seed="


class LabelConfigInfo(NamedTuple):
    from_name: str
    to_name: str
    labels: List[str]
    image_value_key: str  # the `value="$image"` key, minus the $


def parse_label_config(xml: Optional[str], settings: Settings) -> LabelConfigInfo:
    """
    Read the polygon control tag out of a Label Studio labeling config.

    Falls back to the configured `LS_FROM_NAME` / `LS_TO_NAME` when the XML is
    absent or unparseable - Label Studio omits `label_config` on some code
    paths, and a missing config is not a reason to refuse to serve a prediction.
    """
    fallback = LabelConfigInfo(
        from_name=settings.ls_from_name,
        to_name=settings.ls_to_name,
        labels=[],
        image_value_key="image",
    )
    if not xml or not xml.strip():
        return fallback

    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        log.warning("Could not parse label_config, falling back to env tag names: %s", exc)
        return fallback

    control = None
    for tag in ("PolygonLabels", "Polygon"):
        control = root.find(f".//{tag}")
        if control is not None:
            break
    if control is None:
        log.warning(
            "label_config declares no PolygonLabels/Polygon control. This project is not "
            "set up for polygon segmentation; falling back to env tag names."
        )
        return fallback

    from_name = control.get("name") or settings.ls_from_name
    to_name = control.get("toName") or settings.ls_to_name

    labels = [
        child.get("value")
        for child in control.findall("./Label")
        if child.get("value")
    ]

    image_value_key = "image"
    for image in root.findall(".//Image"):
        if image.get("name") == to_name:
            image_value_key = (image.get("value") or "$image").lstrip("$")
            break

    return LabelConfigInfo(from_name, to_name, labels, image_value_key)


# --------------------------------------------------------------------------
# Coordinate conversion
# --------------------------------------------------------------------------

def to_percent(points: Sequence[Sequence[float]], width: float, height: float) -> List[List[float]]:
    """
    Absolute pixels to Label Studio percentages.

    Values are clipped into [0, 100] rather than merely rounded: Label Studio
    drops an out-of-range region entirely, so a vertex a hair past the edge
    would cost the whole pre-annotation.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"image dimensions must be positive, got {width}x{height}")
    out = []
    for x, y in points:
        px = min(100.0, max(0.0, (float(x) / width) * 100.0))
        py = min(100.0, max(0.0, (float(y) / height) * 100.0))
        out.append([round(px, 4), round(py, 4)])
    return out


def from_percent(points: Sequence[Sequence[float]], width: float, height: float) -> List[List[float]]:
    """
    Label Studio percentages back to absolute pixels.

    Not used on the serving path. It is here because Dev 4 and, later, Tier 3
    receive `result` regions in percentage space and have to get back to pixels
    before any IoU is meaningful, and the conversion should have exactly one
    implementation across the two tracks. See `docs/integration-notes.md`.
    """
    return [
        [round(float(x) / 100.0 * width, 4), round(float(y) / 100.0 * height, 4)]
        for x, y in points
    ]


# --------------------------------------------------------------------------
# Prediction assembly
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PredictionBundle:
    prediction: LSPredictionResult
    region_id: str
    label: str
    points_percent: List[List[float]]


def resolve_label(task: QueueTask, info: LabelConfigInfo, settings: Settings) -> str:
    """
    Pick the class label for the region.

    Grounding DINO's text label (`bounding_box.label`, e.g. "car") is the right
    answer, but only if the labeling config declares it - Label Studio discards
    a region carrying an undeclared label. When it is not declared we fall back
    to `LS_LABEL_FALLBACK` and log it, because the alternative is a region that
    vanishes for a reason nobody can see from the UI.
    """
    label = (task.bounding_box.label or "").strip()
    if not label:
        return settings.ls_label_fallback
    if info.labels and label not in info.labels:
        log.warning(
            "Task %s carries label %r, which the labeling config does not declare "
            "(declared: %s). Serving it as %r instead. Add the label to "
            "label_studio/labeling_config.xml to keep the class information.",
            task.task_id, label, info.labels or "<none>", settings.ls_label_fallback,
        )
        return settings.ls_label_fallback
    return label


def build_prediction(
    task: QueueTask,
    wiggled: WiggleResult,
    dims: ImageDims,
    settings: Settings,
    info: LabelConfigInfo,
) -> PredictionBundle:
    """Assemble the single-region `predictions` block for one served task."""
    points_percent = to_percent(wiggled.points, dims.width, dims.height)
    label = resolve_label(task, info, settings)
    region_id = uuid.uuid4().hex[:10]
    seed = wiggled.params.seed

    region: Dict[str, Any] = {
        "id": region_id,
        "from_name": info.from_name,
        "to_name": info.to_name,
        "type": "polygonlabels",
        "original_width": dims.width,
        "original_height": dims.height,
        "image_rotation": 0,
        "value": {
            "points": points_percent,
            "polygonlabels": [label],
            "closed": True,
        },
        # Channel 1 for the seed. `meta.text` is Label Studio's own per-region
        # metadata field, so it survives storage and is readable from the frontend.
        "meta": {
            "text": [
                f"{SEED_META_PREFIX}{seed}",
                f"task_id={task.task_id}",
            ]
        },
    }

    prediction = LSPredictionResult(
        result=[region],
        # Not a confidence. Label Studio sorts and filters predictions by score,
        # and the wiggle is sampled rather than ranked, so a constant keeps the
        # ordering stable instead of implying a calibration we do not have.
        score=0.0,
        # Channel 2 for the seed.
        model_version=encode_model_version(settings.model_version, seed),
    )

    return PredictionBundle(
        prediction=prediction,
        region_id=region_id,
        label=label,
        points_percent=points_percent,
    )


def empty_prediction(settings: Settings) -> LSPredictionResult:
    """
    A prediction with no regions.

    Returned when the queue is empty or the task cannot be wiggled. Label Studio
    requires one `results` entry per requested task, positionally matched, so
    omitting the entry would shift every later task's prediction onto the wrong
    task - far worse than showing a blank canvas.
    """
    return LSPredictionResult(result=[], score=0.0, model_version=settings.model_version)


def encode_model_version(model_version: str, seed: str) -> str:
    return f"{model_version}{MODEL_VERSION_SEED_SEPARATOR}{seed}"


def decode_model_version(model_version: str) -> Optional[str]:
    """Recover a seed from `model_version`. Returns None if it carries no seed."""
    if MODEL_VERSION_SEED_SEPARATOR not in (model_version or ""):
        return None
    return model_version.split(MODEL_VERSION_SEED_SEPARATOR, 1)[1] or None


_SEED_IN_META = re.compile(re.escape(SEED_META_PREFIX) + r"([0-9a-zA-Z_\-]+)")


def seed_from_region_meta(region: Dict[str, Any]) -> Optional[str]:
    """Recover a seed from a region's `meta.text` entries."""
    texts = ((region or {}).get("meta") or {}).get("text") or []
    for entry in texts:
        match = _SEED_IN_META.search(str(entry))
        if match:
            return match.group(1)
    return None


# --------------------------------------------------------------------------
# Task import shape
# --------------------------------------------------------------------------

def task_data_for_ls(task: QueueTask, dims: Optional[ImageDims] = None) -> Dict[str, Any]:
    """
    The `data` dict for a Label Studio task created from a `QueueTask`.

    `task_id` is the field that matters: `LSAnnotationUpdatedPayload.task_id` is
    documented as "our internal QueueTask.task_id, passed via LS task.data", so
    it has to be here or Dev 4's webhook cannot be tied back to anything.

    `width`/`height` are included when known so `/predict` can skip the network
    probe - see `image_meta.ImageDimensionResolver.resolve`.
    """
    data: Dict[str, Any] = {
        "image": task.image_url,
        "task_id": task.task_id,
        "image_id": task.image_id,
        "queue": task.queue.value,
    }
    if dims is not None:
        data["width"] = dims.width
        data["height"] = dims.height
    return data
