# Tier 4 — the learned wiggle policy

What Tier 4 trains, why, and how a checkpoint gets from the trainer to the
annotator's screen. Written after the original team dispersed, when the
Tier 4 worker merged on Sep 6 turned out to be a placeholder (random tensors,
a single `nn.Linear`, and a "weights" file containing a string).

## What is trained

Tier 2's stochastic decoder (`services/serving_ui/app/wiggle.py`) perturbs the
Tier 1 consensus mask with a **5-dimensional affine action**
`(tx, ty, log_scale, theta, normal_offset)`. That vector is the action the
system actually takes, and Tier 2 §2 asks for exactly this low-dimensional
action space so the critic can assign credit (Fault 2).

The policy is a Gaussian over that vector:

```
z ~ N(mu_theta(s), diag(sigma_theta^2))
mu_theta(s) = 3 * tanh((W2 tanh(W1 s + b1) + b2) / 3)
sigma_theta = exp(clip(log_std, -2, 0.5))        # state-independent
```

`s` is an 8-feature observation of the state the decoder acts on
(`common/rl_policy.py::OBS_FEATURES`): the mask's size, aspect ratio, position,
compactness and vertex count relative to the image, plus Tier 1's MCD
`geometric_variance` and `class_logit_entropy`. About 400 parameters.

`wiggle.py` applies `z` exactly as it always applied its standard-normal draw:
`z = mean + std * eps`, where `eps` is still the seeded PCG64 draw. The
initial checkpoint has `mean = 0, std = 1`, so **turning the policy on changes
nothing an annotator sees until a batch has been trained**, and every
pre-policy seed still replays bit-for-bit.

### Why not LoRA on the SAM2 decoder

The spec's long-term picture fine-tunes SAM2's decoder through LoRA adapters.
That is not what was built, for three reasons:

1. **The action Tier 2 emits is these five numbers.** SAM2's decoder output is
   treated as the distribution mean (`mu` = Tier 1's consensus mask); the
   exploration Tier 2 performs, and that humans react to, is the affine wiggle.
   PPO has to learn the distribution it actually samples from.
2. **It trains on a CPU.** A batch of 64 trains in well under a second. LoRA on
   SAM2-Large needs a CUDA GPU with far more than this machine's 2 GB.
3. **It closes the loop now** without touching the frozen contracts. Moving to
   LoRA later means changing what `mu` is, not how rollouts are recorded.

## How rollouts are recorded

On every serve, `serving_ui` records three numbers PPO needs and the polygons
alone do not provide:

| Field | Meaning |
| --- | --- |
| `policy_obs` | the observation `s` |
| `action_z` | the sampled standardised action `z` |
| `action_log_prob` | `log pi_old(z \| s)` under the serving checkpoint |

They travel `ServedWiggleRecord` → `wiggle_cache:{seed}` (Redis) →
`Stage1Output` → `ExperienceTuple` → `tier3.replay_buffer` (three new nullable
columns). All additions are optional fields on Tier 3-owned schemas.

The serving checkpoint's version is stamped as `model_version` on the
Label Studio prediction, the cache entry and the replay-buffer row.

## The training step

`services/tier4_worker/`:

1. Read the serving checkpoint (`models/policy/current.json`).
2. Claim the **oldest 64** unconsumed tuples of exactly that `model_version`
   that carry a policy action — one transaction, all or nothing.
3. Re-score the batch under that checkpoint. If the recorded log-probs do not
   match (max abs diff > 1e-4), the batch was not served by this policy and the
   run is refused (`OffPolicyBatchError`).
4. PPO: each rollout is a single-step episode, so the return is the reward and
   the advantage is `R − V(s)`, normalised. Clipped surrogate (ε = 0.2), up to
   10 epochs of minibatch 16, early stop when approximate KL > 0.03, a small
   entropy bonus, gradient-norm clipping. The critic `V(s)` is a 64-64 MLP that
   persists across batches.
5. Publish: write `<next version>.json` and its critic, then atomically flip
   `current.json`. `serving_ui` checks the pointer on every request, so the
   next served task uses the new policy. Older checkpoints stay on disk;
   rolling back is pointing `current.json` at one.
6. Audit row in `tier3.ppo_training_runs`.

Tuples served under a superseded version are never trained on (PPO is
on-policy; `docs/authority.md` C1).

### Hyperparameters

None of these come from the spec. They are standard PPO values, except
`actor_lr = 3e-3` (10x the usual), chosen because the policy is tiny and a
batch costs an hour of human time. Every one is an env var: `PPO_EPOCHS`,
`PPO_MINIBATCH_SIZE`, `PPO_CLIP_RATIO`, `PPO_ACTOR_LR`, `PPO_CRITIC_LR`,
`PPO_ENTROPY_COEF`, `PPO_MAX_GRAD_NORM`, `PPO_TARGET_KL`, `TIER4_BATCH_SIZE`.

The spec's own unset values are still unset and still provisional: `α`, `β`
(`TIER3_ALPHA=1.0`, `TIER3_BETA=0.3`) and `w1..w3` (`EDRDE_W1_CLICKS` etc.).

## A risk in the reward, worth deciding before real training

With no gold mask, Tier 3 uses the production form
`ΔIoU = 1 − IoU(M_served, M_final)`. That term is **larger the more the human
had to change the mask**, so on its own it rewards serving worse masks. Only the
effort penalty `−β·ΔE_norm` pushes the other way. Which one wins depends on
`α`, `β` and the effort weights — exactly the values the spec leaves unset. If
`α` dominates, PPO will learn to serve masks that are as wrong as annotators
will tolerate. Watch the policy's mean `normal_offset` / `log_scale` and
the mean reward in `tier3.ppo_training_runs`. This is the same question as
Q11 / D10 (what `M_initial` is) and it needs an owner.

## Verifying it

- `services/tier4_worker/tests/test_ppo.py`: the numpy and torch policies agree
  to 1e-9; PPO moves the mean to a known optimum on a synthetic bandit; an
  off-policy batch is refused; publishing flips the pointer and keeps the old
  checkpoint.
- `scripts/simulate_annotators.py`: plays Label Studio and a modelled annotator
  through the real running stack. Its "truth" sits 8 px outside Tier 1's mask,
  so a working loop shows the served mask's error falling batch over batch.
