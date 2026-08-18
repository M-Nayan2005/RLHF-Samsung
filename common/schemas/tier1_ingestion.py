"""
Tier 1 — Grounded SAM2 Pre-Inference Output Contract
Owner: Developer 1 (Pre-Inference & Auto-Labeling Engine)
Consumed by: Developer 2 (Routing/QA), persisted by pre_inference service.

This is the payload written to `tier1_predictions` (Postgres) and handed to
the Dual-Metric Router. Field names are frozen — do not rename without a
version bump and a note in #eng-contracts.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, confloat, conint


class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    label: str = Field(..., description="Class text label from Grounding DINO, e.g. 'car'")
    confidence: confloat(ge=0, le=1)


class PolygonMask(BaseModel):
    """A single closed polygon. Coordinates are absolute pixel space (not normalized)."""
    points: List[List[float]] = Field(
        ..., description="[[x1,y1],[x2,y2],...] closed polygon, min 3 points"
    )
    rle: Optional[str] = Field(
        None, description="Optional COCO-style RLE encoding, base64/ascii string"
    )


class MCDSample(BaseModel):
    """One of the 5x Monte Carlo Dropout forward passes."""
    sample_index: conint(ge=0, le=4)
    mask: PolygonMask
    class_logits: List[float] = Field(
        ..., description="Raw per-class logits for this sample, used for entropy calc"
    )


class GroundedSAM2Output(BaseModel):
    """
    Full Tier 1 record for one (image, prompt) pair after the
    Grounding DINO -> SAM2 5x-MCD pipeline has run.
    """
    image_id: str = Field(..., description="Stable UUID assigned at ingestion")
    image_url: str = Field(..., description="Signed/public URL or storage key the UI can render")
    text_prompt: str = Field(..., description="Prompt fed to Grounding DINO, e.g. 'car'")

    bounding_box: BoundingBox
    mcd_samples: List[MCDSample] = Field(..., min_items=5, max_items=5)

    # Derived metrics — computed by Dev 1's pipeline, consumed by Dev 2's router.
    # These are NOT recomputed downstream; router trusts these values.
    geometric_variance: confloat(ge=0) = Field(
        ..., description="Spatial variance across the 5 MCD polygon samples (e.g. mean pairwise IoU-based spread)"
    )
    class_logit_entropy: confloat(ge=0) = Field(
        ..., description="Entropy of the averaged class logits across MCD samples"
    )
    consensus_mask: PolygonMask = Field(
        ..., description="Mean/representative mask across MCD samples — this is the baseline geometry (M_initial) handed to Tier 2"
    )

    model_version: str = Field(..., description="e.g. 'grounding-dino-1.0+sam2-hiera-l'")
    created_at: str = Field(..., description="ISO-8601 UTC timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "image_id": "img_8f3a1c",
                "image_url": "https://cdn.example.com/raw/img_8f3a1c.jpg",
                "text_prompt": "car",
                "bounding_box": {"x_min": 100, "y_min": 50, "x_max": 400, "y_max": 300, "label": "car", "confidence": 0.94},
                "mcd_samples": [],
                "geometric_variance": 0.032,
                "class_logit_entropy": 0.11,
                "consensus_mask": {"points": [[100, 50], [400, 50], [400, 300], [100, 300]]},
                "model_version": "grounding-dino-1.0+sam2-hiera-l",
                "created_at": "2026-08-18T10:15:00Z",
            }
        }
