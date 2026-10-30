"""
Seed `tier1_predictions` so the router has something to route.

Tier 1 (`pre_inference`) needs a GPU and Grounding DINO + SAM2 weights, so for
local runs this stands in for it.

    python seed_db.py                      # the original 4 routing-branch variants
    python seed_db.py --count 200 --junior # 200 varied, junior-routable objects

`--junior` mode draws a random irregular polygon per row inside the test image
served by serving_ui (tests/assets/test.jpg, 640x480), with low variance and
entropy so the router sends it to junior_queue. That is the queue serving_ui
claims from, and what a training batch needs 64 of.
"""
import argparse
import json
import math
import os
import random
import uuid
from datetime import datetime, timezone
import sys

# Ensure config and db modules can be imported when run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db import init_db, get_db_connection

DEFAULT_IMAGE_URL = "http://localhost:8003/assets/test.jpg"
DEFAULT_IMAGE_SIZE = (640, 480)

INSERT_SQL = """
    INSERT INTO tier1_predictions (
        image_id, image_url, text_prompt, bounding_box, mcd_samples,
        geometric_variance, class_logit_entropy, consensus_mask, model_version, created_at
    ) VALUES (
        %(image_id)s, %(image_url)s, %(text_prompt)s, %(bounding_box)s, %(mcd_samples)s,
        %(geometric_variance)s, %(class_logit_entropy)s, %(consensus_mask)s, %(model_version)s, %(created_at)s
    ) ON CONFLICT (image_id) DO NOTHING;
"""


def _row(mock: dict) -> dict:
    return {
        'image_id': mock['image_id'],
        'image_url': mock['image_url'],
        'text_prompt': mock['text_prompt'],
        'bounding_box': json.dumps(mock['bounding_box']),
        'mcd_samples': json.dumps(mock['mcd_samples']),
        'geometric_variance': mock['geometric_variance'],
        'class_logit_entropy': mock['class_logit_entropy'],
        'consensus_mask': json.dumps(mock['consensus_mask']),
        'model_version': mock['model_version'],
        'created_at': mock['created_at']
    }


def random_object(rng: random.Random, width: int, height: int, image_url: str) -> dict:
    """One plausible Tier 1 output: an irregular blob, its box, and 5 identical MCD samples."""
    rx, ry = rng.uniform(50, 130), rng.uniform(50, 110)
    cx = rng.uniform(rx + 10, width - rx - 10)
    cy = rng.uniform(ry + 10, height - ry - 10)
    n = rng.randint(8, 16)
    phase = rng.uniform(0, 2 * math.pi)
    points = []
    for i in range(n):
        a = phase + 2 * math.pi * i / n
        r = rng.uniform(0.8, 1.1)
        points.append([round(cx + rx * r * math.cos(a), 2), round(cy + ry * r * math.sin(a), 2)])

    xs, ys = [p[0] for p in points], [p[1] for p in points]
    label = rng.choice(["cat", "car", "person", "dog"])
    mask = {"points": points, "rle": None}
    return {
        "image_id": f"img_{uuid.uuid4().hex[:8]}",
        "image_url": image_url,
        "text_prompt": label,
        "bounding_box": {
            "x_min": max(min(xs) - 5, 0), "y_min": max(min(ys) - 5, 0),
            "x_max": min(max(xs) + 5, width), "y_max": min(max(ys) + 5, height),
            "label": label, "confidence": round(rng.uniform(0.8, 0.99), 3),
        },
        "mcd_samples": [
            {"sample_index": i, "mask": mask, "class_logits": [2.0, 0.1]} for i in range(5)
        ],
        "geometric_variance": round(rng.uniform(0.005, 0.04), 4),
        "class_logit_entropy": round(rng.uniform(0.02, 0.15), 4),
        "consensus_mask": mask,
        "model_version": "mock-tier1-seed",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def seed_variants(cursor):
    # Load the mock fixture
    mock_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tests", "mocks", "tier1_output.json")
    with open(mock_path, "r") as f:
        base_mock = json.load(f)

    # Create variants to hit different routing branches
    variants = [
        # Junior route: low variance, low entropy
        {"v": 0.02, "e": 0.1, "desc": "Junior Test"},
        # Senior route (confidently wrong): low variance, high entropy
        {"v": 0.02, "e": 0.4, "desc": "Senior Test (High Entropy)"},
        # Senior route (messy mask): high variance, low entropy
        {"v": 0.15, "e": 0.1, "desc": "Senior Test (High Variance)"},
        # Consensus route: extreme combined (e.g. variance > 0.2 and entropy > 0.5)
        {"v": 0.25, "e": 0.6, "desc": "Consensus Test"},
    ]

    for variant in variants:
        mock = dict(base_mock)
        mock["image_id"] = f"img_{uuid.uuid4().hex[:8]}"
        mock["geometric_variance"] = variant["v"]
        mock["class_logit_entropy"] = variant["e"]
        mock["created_at"] = datetime.now(timezone.utc).isoformat()
        cursor.execute(INSERT_SQL, _row(mock))
        print(f"Inserted variant: {variant['desc']} ({mock['image_id']})")


def seed_junior(cursor, count: int, image_url: str, width: int, height: int, seed: int):
    rng = random.Random(seed)
    for _ in range(count):
        cursor.execute(INSERT_SQL, _row(random_object(rng, width, height, image_url)))
    print(f"Inserted {count} junior-routable predictions for {image_url} ({width}x{height}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=0, help="number of random objects to insert")
    parser.add_argument("--junior", action="store_true", help="random junior-routable objects instead of the 4 variants")
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--width", type=int, default=DEFAULT_IMAGE_SIZE[0])
    parser.add_argument("--height", type=int, default=DEFAULT_IMAGE_SIZE[1])
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducible objects")
    args = parser.parse_args()

    # Initialize DB (creates tier1_predictions placeholder and all queue tables)
    init_db()
    print("Database initialized successfully.")

    conn = get_db_connection()
    cursor = conn.cursor()
    if args.junior:
        seed_junior(cursor, args.count or 64, args.image_url, args.width, args.height, args.seed)
    else:
        seed_variants(cursor)
    conn.commit()
    cursor.close()
    conn.close()
    print("Seed complete. The router picks the rows up within ~5 s.")


if __name__ == "__main__":
    main()
