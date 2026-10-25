"""
Drive the whole loop without a browser: play Label Studio and a simulated annotator.

    python scripts/simulate_annotators.py --annotations 400

For each task it does exactly what Label Studio + the instrumentation script do:

  1. POST serving_ui /predict           - claims a routed task, serves a wiggled mask
  2. POST serving_ui /telemetry/raw     - the effort beacon (serving_ui forwards it to the gateway)
  3. POST webhook_gateway /webhooks/label-studio
                                        - the stock, nested Label Studio webhook

From there the real services take over: gateway -> telemetry:ingest -> Tier 3
reward -> replay buffer -> Tier 4 PPO -> new checkpoint -> serving_ui.

THE ANNOTATOR IS A MODEL, NOT A PERSON. Each object's "true" boundary is
Tier 1's mask pushed outward by --true-offset px (Tier 1 masks that are
systematically too tight). Correcting the served mask costs clicks, cursor
travel and dwell time in proportion to how far it is from the truth. That
gives the loop a known right answer: if PPO works, the policy's mean
normal_offset should drift positive and effort per task should fall. Real
reward data only comes from real annotators on :8081.

Needs: the stack up, and routed tasks (routing_qa/seed_db.py --junior).
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parents[1]
LABEL_CONFIG = (REPO_ROOT / "services" / "serving_ui" / "label_studio" / "labeling_config.xml").read_text(encoding="utf-8")


def env_value(name: str, default: str) -> str:
    """Read a key from the process env, else from the repo's .env, else the default."""
    if os.environ.get(name):
        return os.environ[name]
    env_file = REPO_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == name:
                return value.split(" #", 1)[0].strip().strip('"')
    return default


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def push_out(points, offset):
    """Move every vertex `offset` px away from the polygon's mean point."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    out = []
    for x, y in points:
        d = math.hypot(x - cx, y - cy) or 1.0
        out.append([x + (x - cx) / d * offset, y + (y - cy) / d * offset])
    return out


def effort(served, truth, rng):
    """The annotator model: effort grows with how far each vertex has to be dragged."""
    dists = [math.hypot(a[0] - b[0], a[1] - b[1]) for a, b in zip(served, truth)]
    moved = [d for d in dists if d > 3.0]  # under 3 px nobody bothers
    clicks = 1 + len(moved) + sum(int(d // 15) for d in moved)
    path = sum(d * rng.uniform(1.5, 3.0) for d in moved) + rng.uniform(40, 120)
    dwell = int(1200 + 650 * len(moved) + 25 * sum(moved) + rng.uniform(0, 800))
    return clicks, round(path, 1), dwell, sum(dists) / len(dists)


def to_percent(points, w, h):
    return [[round(x / w * 100.0, 4), round(y / h * 100.0, 4)] for x, y in points]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=int, default=400)
    parser.add_argument("--serving-url", default="http://localhost:8003")
    parser.add_argument("--gateway-url", default="http://localhost:8004")
    parser.add_argument("--true-offset", type=float, default=8.0, help="px the true boundary sits outside Tier 1's mask")
    parser.add_argument("--batch-size", type=int, default=int(env_value("TIER4_BATCH_SIZE", "64")))
    parser.add_argument("--wait-for-trainer", type=float, default=180.0,
                        help="seconds to wait for a new checkpoint after each full batch (0 = never wait)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    secret = env_value("LABEL_STUDIO_WEBHOOK_SECRET", "default_secret")
    client = httpx.Client(timeout=10.0)

    def serving_version() -> str:
        return client.get(f"{args.serving_url}/health").json()["model_version"]

    version = serving_version()
    served_under = 0
    window = []
    print(f"serving checkpoint: {version}")

    done = 0
    while done < args.annotations:
        if args.wait_for_trainer and served_under >= args.batch_size:
            print(f"  {served_under} served under {version}; waiting for Tier 4 to publish the next checkpoint...")
            deadline = time.time() + args.wait_for_trainer
            while serving_version() == version and time.time() < deadline:
                time.sleep(2)
            new_version = serving_version()
            if new_version == version:
                print("  no new checkpoint arrived; carrying on under the same version")
            else:
                print(f"serving checkpoint: {new_version}")
            version, served_under = new_version, 0

        ls_task_id = 100000 + done
        pred = client.post(f"{args.serving_url}/predict", json={
            "tasks": [{"id": ls_task_id, "data": {}}], "label_config": LABEL_CONFIG,
        }).json()
        result = pred["results"][0]
        if not result.get("result"):
            print("serving_ui returned an empty prediction - out of routed tasks? "
                  "Seed more: docker compose exec routing_qa python seed_db.py --junior --count 200")
            return 1
        seed = result["model_version"].split("|seed=", 1)[1]
        served = client.get(f"{args.serving_url}/served/by-seed/{seed}").json()
        region = result["result"][0]
        w, h = served["image_width"], served["image_height"]

        truth = push_out(served["baseline_points"], args.true_offset)
        truth = [[min(max(x + rng.gauss(0, 0.7), 0), w), min(max(y + rng.gauss(0, 0.7), 0), h)] for x, y in truth]
        clicks, path, dwell, mean_err = effort(served["wiggled_points"], truth, rng)
        annotation_id = 500000 + done

        client.post(f"{args.serving_url}/telemetry/raw", json={
            "task_id": served["task_id"],
            "annotation_id": str(annotation_id),
            "client_session_id": f"sim_{uuid.uuid4().hex[:8]}",
            "transport": "xhr",
            "project_id": "1",
            "completed_by": "sim_annotator",
            "lead_time": dwell / 1000.0,
            "wiggle_seed": seed,
            "effort_telemetry": {
                "click_count": clicks, "cursor_path_length_px": path,
                "dwell_time_ms": dwell, "wiggle_seed": seed,
            },
        }).raise_for_status()

        hook = client.post(
            f"{args.gateway_url}/webhooks/label-studio",
            headers={"X-Label-Studio-Secret": secret},
            json={
                "action": "ANNOTATION_CREATED",
                "annotation": {
                    "id": annotation_id,
                    "completed_by": 1,
                    "lead_time": dwell / 1000.0,
                    "created_at": now_iso(),
                    "updated_at": now_iso(),
                    "task": ls_task_id,
                    "result": [{
                        "id": region["id"],
                        "type": "polygonlabels",
                        "from_name": region["from_name"],
                        "to_name": region["to_name"],
                        "original_width": w,
                        "original_height": h,
                        "value": {
                            "points": to_percent(truth, w, h),
                            "polygonlabels": region["value"]["polygonlabels"],
                            "closed": True,
                        },
                        "meta": region.get("meta", {}),
                    }],
                },
                "task": {"id": ls_task_id, "project": 1, "data": {"task_id": served["task_id"]}},
                "project": {"id": 1},
            },
        )
        if hook.status_code != 200:
            print(f"gateway rejected the webhook: {hook.status_code} {hook.text[:300]}")
            return 1

        done += 1
        served_under += 1
        window.append(mean_err)
        if done % 32 == 0:
            recent = window[-32:]
            print(f"  {done:4d} annotated | last 32: mean vertex error {sum(recent) / len(recent):5.2f} px "
                  f"(served under {version})")

    print(f"done: {done} simulated annotations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
