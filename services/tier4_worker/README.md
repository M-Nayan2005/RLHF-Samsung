# Tier 4: PPO training for the wiggle policy

Waits for 64 on-policy experience tuples in `tier3.replay_buffer`, runs one PPO
update on the Tier 2 wiggle policy, and publishes the result as the next
checkpoint. `serving_ui` picks it up on its next request. Design and
rationale: [`docs/tier4-policy.md`](../../docs/tier4-policy.md).

| File | Role |
| --- | --- |
| `trainer.py` | The poll → claim → train → publish → audit loop. `python -m tier4_worker.trainer` |
| `dataset.py` | Claims the oldest N tuples of one `model_version`, atomically; writes `tier3.ppo_training_runs` |
| `ppo.py` | Torch mirror of `common/rl_policy.py`, the critic, and the clipped-surrogate update |
| `deploy.py` | Writes the checkpoint + critic to `models/policy/`, flips `current.json`, notifies `policy_updates` |
| `seed_db.py` | Synthetic trainable tuples, for exercising this worker on its own |

## Running

```bash
docker compose up -d --build tier4_worker          # with the rest of the stack
docker compose logs -f tier4_worker                # "Waiting for a batch: n/64 ..."

# On its own, with synthetic data:
docker compose run --rm tier4_worker python -m tier4_worker.seed_db --count 65

# Tests (CPU torch needed locally: pip install torch --index-url https://download.pytorch.org/whl/cpu)
cd services/tier4_worker && python -m pytest
```

## Checkpoints

`models/policy/` (bind-mounted into `serving_ui` and this worker):

```
current.json                            {"version": "..."} - what is being served
serving-ui-stochastic-0.1.0.json        the identity policy (the original fixed wiggle)
serving-ui-stochastic-0.1.1.json        after the first trained batch
serving-ui-stochastic-0.1.1.critic.pt   the value network, carried into the next batch
```

Roll back by pointing `current.json` at an older version. Tuples served under
the version you leave are never trained on.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `TIER4_BATCH_SIZE` | 64 | the plan's batch size |
| `TIER4_POLL_INTERVAL_S` | 5 | |
| `SERVING_UI_MODEL_VERSION` | `serving-ui-stochastic-0.1.0` | name of the bootstrap checkpoint; must match serving_ui |
| `PPO_EPOCHS` / `PPO_MINIBATCH_SIZE` | 10 / 16 | |
| `PPO_CLIP_RATIO` | 0.2 | |
| `PPO_ACTOR_LR` / `PPO_CRITIC_LR` | 3e-3 / 1e-3 | |
| `PPO_ENTROPY_COEF` | 1e-3 | |
| `PPO_TARGET_KL` | 0.03 | early stop within a batch |

None of these are spec values; see the hyperparameter note in the design doc.
