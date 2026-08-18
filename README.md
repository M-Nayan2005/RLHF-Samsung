# 26TS09VITV_AI_enabled_Object_Segmentation_using_Reinforcement_Learning_with_Human_Feedback
SRIB-PRISM Program

## RLHF Segmentation Pipeline — Tier 1 & Tier 2 Scaffold

Start here: **`IMPLEMENTATION_PLAN.md`** — contracts, work breakdown across 4
developers, ready-to-paste AI coding prompts, and the git/integration
workflow for tonight's build.

Quick start:
```bash
cp .env.example .env      # fill in checkpoint paths / secrets
docker compose up postgres redis
# each dev then runs their own service, see IMPLEMENTATION_PLAN.md §2
```

Frozen contracts live in `common/schemas/` — read before writing any service code.
Fixture payloads for local testing without teammates live in `tests/mocks/`.
