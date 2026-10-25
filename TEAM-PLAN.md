# RLHF-Samsung: team plan

**Last updated:** 2026-10-26 · **Repo:** https://github.com/RayyanAzeez/RLHF-Samsung · **Branch:** `main`

This is the starting point for everyone on the team, and for any AI agent working
on this repo. It says where the code lives now, what changed in the last
integration pass, and who does what next. Read it before anything else, then
read `HANDOFF.md`.

---

## 1. TL;DR

- The repo has moved. The original `Quirrells-Quill/RLHF-Samsung` was deleted.
  The project now lives at **`RayyanAzeez/RLHF-Samsung`**, on `main`.
- **The whole pipeline now runs end to end** in Docker, for the first time:
  Tier 1 (seeded stand-in) → routing → serving → Label Studio/webhook → reward →
  PPO training → new policy served back, with no restart.
- The merged code on Sep 6 could not run: the compose file did not parse, Tier 3
  and Tier 4 crashed on startup, and several handoffs between services silently
  dropped every annotation. All of that is fixed.
- The old Tier 4 was a placeholder that trained nothing. It is now a real PPO
  trainer. It was verified on a simulated annotator over six batches, and the
  policy learned in the right direction.
- **Not done yet:** real images through Tier 1 on a GPU, real people annotating,
  and a decision on the reward weights. The lab's GPUs make the first two
  possible. The third is a team decision, and it must come before any real
  training (section 5).

---

## 2. Where the repo is now

| | |
| --- | --- |
| **Canonical repo** | `https://github.com/RayyanAzeez/RLHF-Samsung` |
| **Branch to work from** | `main` (currently `aa7b7a9`) |
| Old repo | `Quirrells-Quill/RLHF-Samsung`: **deleted**, every request returns 404 |
| Stale copy | `M-Nayan2005/RLHF-Samsung`: stops at `f750fc8` (Sep 6), before every fix below. Do not build on it. |

**If you already have a clone**, repoint it:

```bash
git remote set-url origin https://github.com/RayyanAzeez/RLHF-Samsung.git
git fetch --prune origin
git checkout main && git reset --hard origin/main   # only if you have no local work to keep
```

**If you don't have a clone:**

```bash
git clone https://github.com/RayyanAzeez/RLHF-Samsung.git
```

**Workflow.** Rayyan will add everyone as a collaborator. Until then, fork the
repo and open PRs from your fork. Either way:

1. Branch off `main`: `feature/<area>-<short-name>` or `fix/<short-name>`.
2. Run the test suites (section 4) before opening a PR.
3. Open a PR into `main`. Don't push to `main` directly.
4. Update `HANDOFF.md` when you finish a piece of work.

---

## 3. What changed in the last integration pass (commit `aa7b7a9`)

### 3.1 Bugs that stopped the pipeline from running

| Problem on the Sep 6 `main` | Fix |
| --- | --- |
| `docker-compose.yml` did not parse (`tier4_worker` was indented under `volumes:`) | Rewritten. Every service uses the in-stack Postgres and Redis. |
| Nothing wrote the Redis key `wiggle_cache:{seed}`, so Tier 3 dropped **every** annotation | `services/serving_ui/app/wiggle_cache.py` writes it on every serve |
| `serving_ui` asked `routing_qa` for queue `junior`, but `routing_qa` only accepts `junior_queue`, so every real serve came back empty | Default and `.env.example` corrected |
| `webhook_gateway` image could not build (wrong context), and its volume mount hid its own code | Fixed in compose |
| Gateway only accepted the flat test payload, not the nested body stock Label Studio sends | `flatten_ls_webhook()` in `services/webhook_gateway/main.py` |
| Tier 3 consumer imported `services.tier3_worker.*`, which doesn't exist inside the container; read the wrong DB variable; queried its tables before creating them | Package-relative imports, `TIER3_PG_DSN`, schema applied at startup |
| An annotation whose effort beacon never arrived was rewarded as **zero effort** | Stage 1 drops it (`reason=no_telemetry`) |
| `routing_qa` never created its own tables unless someone seeded by hand | `init_db()` at startup, with retry |
| Tier 4 crashed on import, and its training step was fake: random tensors, a single `nn.Linear`, and a "weights" file containing a text string. It stalled for good after one batch. | Rewritten: section 3.2 |

### 3.2 New components

- **`common/rl_policy.py`**: the trainable policy, shared by Tier 2 and Tier 4.
  It is a small Gaussian policy over the **5-number affine wiggle** that
  `serving_ui` applies to Tier 1's mask: translate x/y, scale, rotate, and push
  the boundary out or in. The input is 8 features of the mask, plus Tier 1's
  uncertainty scores. About 400 parameters. The starting checkpoint reproduces
  the old fixed wiggle exactly.
- **Tier 4 (`services/tier4_worker/`)**: real PPO.
  1. Claims the oldest 64 tuples served by the current policy version.
  2. Checks they really came from that version, and refuses the batch if not.
  3. Trains with clipped PPO plus a value-network baseline.
  4. Publishes the next version to `models/policy/`.
  5. Logs the run in `tier3.ppo_training_runs`.

  Rolling back means pointing `models/policy/current.json` at an older version.
- **`serving_ui`** samples from whichever checkpoint is current, reloading it
  without a restart, and records what PPO needs: the observation, the action,
  and the action's log-probability.
- **Contract additions.** `WiggleCacheEntry`, `Stage1Output` and
  `ExperienceTuple` (all in the Tier 3-owned `common/schemas/tier3_rlhf.py`)
  gained optional `policy_obs` / `action_z` / `action_log_prob` fields.
  `tier3.replay_buffer` has three matching nullable columns. No frozen Tier 1/2
  contract was edited.
- **`scripts/simulate_annotators.py`**: plays both Label Studio and a modelled
  annotator against the running stack, so the loop can be tested without a
  browser. **The annotator is a model, not real data.**
- **`services/routing_qa/seed_db.py --junior --count N`**: stands in for Tier 1
  by writing routable predictions straight into `tier1_predictions`.
- **`docs/tier4-policy.md`**: what is trained, why it isn't LoRA on SAM2 yet,
  and the reward risk below.

### 3.3 How it was verified

- **Tests:** serving_ui 119, tier3_worker 74, tier4_worker 5 (including "PPO
  finds a known optimum" and "an off-policy batch is refused"). All pass.
- **Live run:** 420 simulated annotations in Docker trained six PPO batches,
  versions `0.1.0` → `0.1.6`, each served live.
  - The policy's mean boundary push rose every batch, 0 → 0.68, toward the
    simulated truth.
  - The annotator's average correction fell from ~10.8 px to ~7.1 px.
  - Zero errors and zero dropped annotations in any service log.
- **Not verified:** a real person annotating in a browser; Tier 1 on a GPU.

---

## 4. Running it

Prerequisites: Docker Desktop, Python 3.11.

```bash
cp .env.example .env
docker compose up -d --build                          # all services except Tier 1
docker compose exec routing_qa python seed_db.py --junior --count 200
python scripts/simulate_annotators.py --annotations 200   # or annotate at http://localhost:8081
docker compose logs -f tier4_worker                   # one PPO update per 64 annotations
docker compose down                                   # when finished
```

**Annotate on `:8081`, never `:8080`.** On `:8080` everything looks like it
works, but no effort telemetry is captured.

**Tests:**

```bash
cd services/serving_ui && python -m pytest
cd services/tier4_worker && python -m pytest          # needs CPU torch
# Tier 3, from the repo root (on Linux/macOS use ':' instead of ';'):
PYTHONPATH="$PWD;$PWD/services" python -m pytest services/tier3_worker
```

---

## 5. Open decisions: these block real training

1. **The reward balance (most important).** The reward is
   `R = α·ΔIoU − β·ΔE_norm`. With no gold mask, `ΔIoU = 1 − IoU(served, final)`,
   which is **larger the more the annotator had to fix**. On its own, that term
   rewards serving worse masks; only the effort penalty pulls back the other way.
   - `α`, `β` and the effort weights `w1..w3` are unset in the spec. The current
     values (`α=1.0`, `β=0.3`, `w=1.0/0.01/0.001`) are placeholders.
   - In the simulation effort won. With real annotators that isn't guaranteed.
   - These get decided from a pilot (section 6, phase 2), not by guesswork.
2. **What the reward is measured against** (open question Q11, divergence D10):
   the consensus mask, or the wiggled mask that was actually served? The default
   is the wiggled one (`TIER3_REFERENCE_MASK=wiggled`). This needs a decision
   before Tier 3 numbers are trusted.
3. **Routing thresholds** (`VARIANCE_THRESHOLD`, `ENTROPY_THRESHOLD`) are also
   placeholders.

**Rule for everyone, and every agent: never pick a value for any of these
yourself.** Raise it with the team, or log it in `docs/reference/open-questions.md`.

---

## 6. The plan

### Phase 1: run on real data (target: about 1 week, all streams in parallel)

#### Stream A: Tier 1 on a lab GPU · **X** (with **Nayan** for the lab machine)

Tier 1 (`services/pre_inference/`) is real code that has never run in this stack.
It only ships as a Modal app, whose account belonged to a former member.

- [ ] CUDA Dockerfile for `pre_inference`, with Grounding DINO + SAM2-hiera-large
      weights. Remove the `tier1` compose profile once it builds.
- [ ] Point it at the stack's Postgres, so it writes `tier1_predictions` the same
      way `routing_qa/seed_db.py` does today.
- [ ] An ingestion path for a real image set: a folder or bucket, plus prompts.
- [ ] Image hosting that **both** the annotators' browsers and `serving_ui`'s
      dimension probe can reach. Today the only image is
      `localhost:8003/assets/test.jpg`.
- **Done when:** real images flow Tier 1 → `routing_qa` → `serving_ui`, and a
  served mask lines up with the object in Label Studio.

#### Stream B: Label Studio and the gateway on the lab machine · **Nayan**

- [ ] Deploy the stack on the lab machine. Label Studio on `:8081` must be
      reachable by annotators.
- [ ] Run `services/serving_ui/label_studio/setup_project.py`: project, labeling
      config, ML backend, and webhook with `LABEL_STUDIO_WEBHOOK_SECRET`.
- [ ] One Label Studio account per annotator.
- [ ] Basic monitoring: queue depth of `telemetry:ingest`, Tier 3 drop reasons,
      Tier 4 "waiting for a batch: n/64".
- **Done when:** a real annotation submitted on `:8081` produces a row in
  `tier3.replay_buffer` that has `action_z` set.

#### Stream C: QA and annotator identity · **Ramyash**

- [ ] Real honeypots. `routing_qa/router.py` injects a 10×10 placeholder gold
      mask. Gold masks need to come from real data, and reach Tier 3 so it can use
      the strict ΔIoU (`WiggleCacheEntry.m_gold`). Note that `serving_ui` only
      ever sees the stripped task, so this needs a server-side path.
- [ ] Implement `/tasks/{id}/requeue` and `/tasks/{id}/honeypot-result`. Both are
      stubs today.
- [ ] Annotator identity: every annotator currently shows up as
      `annotator_unknown` (divergence D16), which makes trust scores meaningless.
- **Done when:** a honeypot round trip updates `annotator_trust_scores` for a
  named annotator.

#### Stream D: browser telemetry check, then CI · **Khushagra**

- [ ] Annotate a few tasks on `:8081` with `RLHF_TELEMETRY_DEBUG=true`. Confirm
      in the browser console and `docker compose logs serving_ui` that beacons
      arrive with sensible click, path and dwell numbers. Also confirm the second,
      `supersedes_prior_beacon` beacon fires with a real `annotation_id`.
- [ ] Write a one-page annotator guide: how to log in on `:8081`, what to do,
      what not to do.
- [ ] GitHub Actions: run the three test suites and `docker compose config -q`
      on every PR.
- **Done when:** CI is green on `main`, and the telemetry check is written up in
  `HANDOFF.md`.

#### Stream E: repo, docs, integration · **Rayyan**

- [ ] Add collaborators; protect `main` (PR + green CI required).
- [ ] **Port the spec docs into the repo.** `docs/authority.md`,
      `docs/reference/{equations,open-questions,glossary,tech-stack}.md`,
      `docs/implementation/spec-alignment.md` and `docs/architecture/` currently
      live only in a local workspace. The code and docs link to them 25 times.
      Check whether the Samsung spec material is OK to have in a **public** repo
      first, and make the repo private if not.
- [ ] Refresh `AGENTS.md`. It still describes only Tiers 1–2.
- [ ] Review every PR that touches `common/`.

### Phase 2: calibrate the reward (whole team, led by Rayyan; ~1 week after phase 1)

1. Pause the trainer: `docker compose stop tier4_worker`.
2. Collect 2–3 batches (128–192 annotations) of real annotation.
3. From `tier3.replay_buffer`, compare the spread of `delta_iou` and
   `delta_e_norm`, and check the effort weights against real click, path and
   dwell numbers.
4. Decide `α`, `β`, `w1..w3` and Q11 together, with the supervisor, and write the
   decision into `docs/reference/open-questions.md`.
5. Clear the pilot tuples, or bump the policy version, so they aren't trained on
   under the old weights.

### Phase 3: train

- **Real training runs on the current policy · Nayan.** Restart `tier4_worker`.
  Watch `tier3.ppo_training_runs`, and the policy's mean and spread per action
  dimension, every batch. Stop it if the masks start getting worse. Rollback is
  in `services/tier4_worker/README.md`.
- **SAM2 LoRA fine-tuning · Nayan + X.** This is the spec's end goal. Today
  Tier 4 learns only the wiggle on top of SAM2's mask, while SAM2 itself is
  frozen. Making SAM2 learn means:
  - LoRA adapters on the SAM2 mask decoder, trained on the lab GPUs;
  - recording whatever the decoder needs at serve time (prompt, image embedding
    reference);
  - Tier 1 loading new SAM2 checkpoints. The Redis channel `policy_updates`
    already exists for that signal.

  Start the design once real data is flowing. Read `docs/tier4-policy.md` first.

### What to ask the lab for

- 1 GPU for Tier 1 inference (16 GB is enough), and one with **24 GB+** for
  SAM2 LoRA training.
- Docker with the NVIDIA container toolkit.
- Persistent storage for Postgres, images and `models/`.
- A port annotators can reach (`8081`), and ideally a hostname.

---

## 7. Rules for contributors and their AI agents

- **Read first:** this file → `HANDOFF.md` → `README.md` →
  `docs/tier4-policy.md` → the decision log for the service you're touching
  (`docs/dev3-decisions.md`, `docs/tier3-dev2-decisions.md`,
  `docs/integration-notes.md`).
- **Frozen contracts:** `common/schemas/tier1_ingestion.py`,
  `routing_queue.py`, `label_studio_webhook.py` and `redis_event.py` are not
  edited without team agreement. `tier3_rlhf.py` accepts *optional, additive*
  fields only.
- **Never invent a hyperparameter the spec leaves unset** (`α`, `β`, `w1..w3`,
  routing thresholds). Ask.
- **Rejected designs, do not reintroduce:** CVAT, neural reward models, a pure
  IoU reward, experience replay (the `replay_buffer` table is on-policy despite
  its name), deterministic decoding, pixel-grid action spaces, synchronous
  webhooks.
- **Don't claim something works without running it.** Every runtime bug above
  passed its unit tests.
- **Windows:** `PYTHONPATH` uses `;`. In Git Bash, `docker exec -w /app/...`
  needs `MSYS_NO_PATHCONV=1`.
- **Update `HANDOFF.md`** before you stop working: what changed, what's verified,
  and what's next.
