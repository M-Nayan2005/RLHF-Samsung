# HANDOFF

Use this document to persist state between agent sessions. Any agent shutting down or yielding must update this file. Any agent waking up must read this file first.

## Current Context (2026-10-26)

The original GitHub repo (`Quirrells-Quill/RLHF-Samsung`) was deleted. The
project now lives at `RayyanAzeez/RLHF-Samsung`, on `main`.
`M-Nayan2005/RLHF-Samsung` is a stale copy from Sep 6 (f750fc8). Team plan and
work split: `TEAM-PLAN.md`.

Branch `fix/cohesive-pipeline` makes the whole loop run end to end for the
first time. Verified in Docker with the annotator simulator: 420 annotations,
six PPO batches, checkpoints 0.1.0 -> 0.1.6 published and served without a
restart, the policy's mean `normal_offset` rising monotonically 0 -> 0.68
toward the simulated truth, and the simulated correction error falling
~10.8 px -> ~7.1 px. Zero errors or dropped annotations in any service log.

### What was broken on `main` and is fixed here

| Problem | Fix |
| --- | --- |
| `docker-compose.yml` did not parse (`tier4_worker` under `volumes:`) | rewritten; all services wired to in-stack Postgres/Redis |
| Nothing wrote `wiggle_cache:{seed}`, so Tier 3 dropped every annotation | `serving_ui/app/wiggle_cache.py`, written on every serve |
| serving_ui asked routing_qa for queue `junior`; routing_qa accepts only `junior_queue` (400 on every real serve) | default + `.env.example` corrected |
| webhook_gateway build context wrong; its volume hid its own code | compose build context = repo root; mount fixed |
| Gateway accepted only the flat contract, not Label Studio's nested webhook | `flatten_ls_webhook` adapter |
| Tier 3 consumer imported `services.tier3_worker.*` (absent in the container); used `DATABASE_URL`, not Tier 3's DSN; queried tier3 tables before creating them | relative imports, `TIER3_PG_DSN`, schema applied at start |
| Beacon-less annotations were rewarded as zero effort | Stage 1 drops them (`reason=no_telemetry`) |
| routing_qa never created its tables unless seeded by hand | `init_db` at startup, with retry |
| Tier 4 was a mock (random tensors, one Linear layer, a string as "weights", nobody consumed the reload signal, stalled after one batch) | real PPO on the 5-d affine wiggle policy; see `docs/tier4-policy.md` |

## Track Status

- Tier 1 `pre_inference`: real code, Modal-only, needs a GPU. Not run here. Stand-in: `routing_qa/seed_db.py --junior`.
- Tier 2 `routing_qa`, `serving_ui`: working; serving_ui serves the learned policy.
- Gateway: working with stock Label Studio webhooks and the beacon join.
- Tier 3 `tier3_worker`: working.
- Tier 4 `tier4_worker`: working PPO, verified on a bandit test and the live loop.

Tests: serving_ui 119, tier3_worker 74, tier4_worker 5 - all passing.

## Next Steps

1. **Decide the reward balance before real training** - see "A risk in the
   reward" in `docs/tier4-policy.md`. `α`, `β`, `w1..w3` are still provisional.
2. Annotate for real through Label Studio on :8081 (needs an LS account + API
   token for `services/serving_ui/label_studio/setup_project.py`). Never verified
   with a human in the browser on this branch.
3. Tier 1 on a GPU (Modal account / secrets were a teammate's), or a Dockerfile for it.
4. Honeypot gold masks in routing_qa are a 10x10 placeholder; Tier 3 never
   receives them (serving_ui sees the stripped task), so it uses the production ΔIoU.
