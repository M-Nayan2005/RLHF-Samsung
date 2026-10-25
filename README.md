# 26TS09VITV_AI_enabled_Object_Segmentation_using_Reinforcement_Learning_with_Human_Feedback
SRIB-PRISM Program

## RLHF Segmentation Pipeline

A closed loop: Tier 1 proposes a mask, Tier 2 serves a deliberately perturbed
("wiggled") version of it to an annotator and measures what correcting it costs,
Tier 3 turns that into a reward, and Tier 4 trains the wiggle policy with PPO
and publishes the new policy back to Tier 2.

```
routing_qa -> serving_ui -> Label Studio (:8081) -> webhook_gateway
  -> telemetry:ingest -> tier3_worker -> tier3.replay_buffer
  -> tier4_worker (PPO) -> models/policy/ -> serving_ui
```

### Run the loop

```bash
cp .env.example .env
docker compose up -d --build                     # everything except Tier 1 (GPU, see below)

# Tier 1 stand-in: write routable predictions straight into tier1_predictions
docker compose exec routing_qa python seed_db.py --junior --count 200

# Either annotate for real at http://localhost:8081 (NOT :8080 - no telemetry there;
# first-time setup: services/serving_ui/README.md), or drive it with the simulator:
python scripts/simulate_annotators.py --annotations 400

docker compose logs -f tier4_worker              # one PPO update per 64 annotations
```

`pre_inference` (Grounding DINO + SAM2) needs a CUDA GPU and deploys on Modal
(`services/pre_inference/modal_app.py`); it is behind the `tier1` compose
profile and has no Dockerfile.

### Where to read

- `docs/tier4-policy.md` - what is trained, how rollouts are recorded, and a reward risk to decide on
- `docs/integration-notes.md` - the Tier 2 / gateway seam
- `docs/tier3-dev2-decisions.md`, `docs/dev3-decisions.md` - per-tier judgment calls
- `HANDOFF.md` - current state

Frozen contracts live in `common/schemas/`. The shared policy is `common/rl_policy.py`.
