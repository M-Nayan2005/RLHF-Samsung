# serving_ui — Tier 2 Interactive Serving & Stochastic Policy

**Developer 3.** The Label Studio ML Backend, the stochastic mask decoder, and the
frontend instrumentation that captures annotator effort.

This is the only tier a human ever sees. It takes a task from Dev 2's router, samples a
deliberately-slightly-wrong mask from the policy, puts it in front of a junior annotator,
and measures what fixing it cost them. That cost is the reward signal the rest of the
system trains on.

---

## The one thing Dev 4 needs to read

**Effort telemetry does not arrive inside the Label Studio webhook. It arrives separately,
as a `POST /telemetry/raw` from this service.** The two have to be joined on
`wiggle_seed` (preferred) or `task_id`.

The plan left this open (§0) and asked Dev 3 to decide. The decision, the reasoning, and
the exact join rule are in [`docs/integration-notes.md`](../../docs/integration-notes.md),
which is written for Dev 4 specifically. The short version is below under
[Telemetry transport](#telemetry-transport).

---

## What it does

| Step | Where |
| --- | --- |
| Label Studio opens a task and calls `POST /predict` | `app/main.py` |
| Claim a `QueueTask` from `routing_qa` (or the fixture in `MOCK_MODE`) | `app/task_source.py` |
| Work out the image's pixel dimensions | `app/image_meta.py` |
| Sample an action from the Gaussian policy and apply it to `baseline_mask` | `app/wiggle.py` |
| Persist the polygon actually served — the action `A_t` | `app/store.py` |
| Convert to a Label Studio `predictions` block in percentage coordinates | `app/ls_format.py` |
| Browser measures clicks, cursor path, boundary dwell | `label_studio/instrumentation/effort_telemetry.js` |
| Beacon fires just before submit; this service enriches and forwards it to Dev 4 | `app/telemetry.py` |

---

## Quick start

### Standalone, no teammates, no Docker

```bash
pip install -r services/serving_ui/requirements.txt

MOCK_MODE=true \
MOCK_TASK_PATH=tests/mocks/routing_task.dev3.json \
SERVED_STORE_PATH=./data/served_wiggles.jsonl \
python -m uvicorn app.main:app --app-dir services/serving_ui --port 8003
```

`MOCK_MODE=true` serves a fixture instead of calling `routing_qa`. Two are provided:

| Fixture | `image_url` | Notes |
| --- | --- | --- |
| `routing_task.dev3.json` | Supabase public object URL | **Use this one.** A real, reachable 640×480 photo. Verified: loads in the browser and probes correctly over HTTPS. Baseline mask sits over the left-hand cat. |
| *(offline)* | `http://localhost:8003/assets/test.jpg` | Same image, served by this service off disk. Swap `image_url` to this when working without network. |
| `routing_task.json` | `cdn.example.com` | The shared fixture from the scaffold. Not reachable — needs `IMAGE_DIM_SOURCE=fixed DEFAULT_IMAGE_WIDTH=800 DEFAULT_IMAGE_HEIGHT=600`. |

Drop any image into `tests/assets/` to serve it at `/assets/<name>`. Note that
`https://images.cocodataset.org/...` **fails TLS certificate verification** in both `httpx`
and `curl`, so a public COCO URL will not probe — keep a local copy instead.

Verify:

```bash
curl -s localhost:8003/health | python -m json.tool

# A full prediction, exactly as Label Studio would ask for it
curl -s -X POST localhost:8003/predict \
     -H 'Content-Type: application/json' \
     -d @tests/mocks/ls_predict_request.json | python -m json.tool

# Calibrate sigma by looking at the result rather than guessing
curl -s 'localhost:8003/wiggle/preview?sigma=0.02' | python -m json.tool
```

### Tests

```bash
python -m pytest          # 113 tests, no network, no teammates, no Docker
```

### Full stack

```bash
cp .env.example .env       # then set LABEL_STUDIO_API_TOKEN
docker compose up --build

python services/serving_ui/label_studio/setup_project.py --import-mock
```

Then open **`http://localhost:8081`** — the proxy, *not* `:8080`. See
[Loading the instrumentation script](#loading-the-instrumentation-script).

---

## The stochastic policy

Tier 2's spec (§2) requires the SAM2 mask decoder to behave as an RL policy rather than a
deterministic function: it samples from `N(mu, sigma^2)`, where `mu` is the model's optimal
guess and `sigma^2` is noise injected on purpose. The wiggled mask is deliberately slightly
wrong, and that is the exploration mechanism — a decoder with no variance generates no
variance to learn from (**Fault 3**).

Here, `mu` is `QueueTask.baseline_mask` (Tier 1's `consensus_mask`) and `sigma` is
`WIGGLE_SIGMA`.

### Which perturbation — and why

The brief asks for an explicit choice between perturbing each vertex independently and
applying a shared jitter. **The default is a shared 5-dimensional affine action**
(`WIGGLE_MODE=affine`):

| Dimension | Effect |
| --- | --- |
| `tx`, `ty` | translation |
| `log_scale` | isotropic dilation, exponentiated so the polygon can never invert |
| `theta` | rotation about the area-weighted centroid |
| `normal_offset` | uniform push along the outward boundary normal |

Independent per-vertex noise would make the action `2N`-dimensional — 400 numbers for a
200-vertex polygon — which is precisely the credit-assignment problem the spec avoids by
insisting the action space is *"low-dimensional latent points and boxes, never a raw pixel
grid"* (**Fault 2**). It also looks wrong in a different way: jagged rather than a
near-miss, so annotators redraw instead of correct.

`WIGGLE_MODE=vertex` implements the alternative for A/B comparison. It is not the default.

The `normal_offset` dimension is the one that makes this more than a rigid-body move —
translation, rotation and scale can only *relocate* a shape, while the offset dilates the
boundary itself, which is the error annotators actually correct.

### What `sigma` is measured against

`WIGGLE_SIGMA=0.02` is dimensionless and the spec never says what it scales. Multiplying
raw pixel coordinates by it would make the perturbation depend on where the object sits in
the frame, which is meaningless. So sigma is a fraction of a length intrinsic to the object
— by default the diagonal of Grounding DINO's bounding box (`WIGGLE_SCALE_REFERENCE`).
`sigma=0.02` then reads as *"about 2% of the object's diagonal, per action dimension"*.

This interpretation is **this track's choice, not the spec's** — recorded as D14 / Q14.

Measured on the shared fixture (a 300×250 box, `sigma=0.02`): mean vertex displacement
**≈21 px**, IoU against the baseline **≈0.83**. Visible, correctable, still recognisably
the same object.

### Reproducibility contract

`LSTelemetryMeta.wiggle_seed` is the only trace of the wiggled polygon that the frozen
contracts carry, so Tier 3 can recover the sampled action `A_t` only if the perturbation
replays exactly. Four things are pinned to make that true, and `test_wiggle.py` asserts
each one:

1. **RNG** — numpy `Generator(PCG64)`. numpy guarantees stream stability for `Generator`
   across versions; `RandomState` and Python's `random` promise less.
2. **Seed derivation** — BLAKE2b-128 of the UTF-8 seed string, big-endian. Any string works,
   including non-hex seeds like the fixtures' `seed_7781`.
3. **Draw order** — one `standard_normal(5)`, documented, append-only. Inserting a draw
   invalidates every seed ever issued.
4. **Vertex order** — `geometry.canonicalize` drops a duplicated closing vertex, forces
   counter-clockwise winding, and rotates the ring to start at the lexicographically
   smallest vertex. An upstream service that flips its winding still gets the same wiggle.

Changing any of them is a breaking change to issued seeds: bump `WIGGLE_ALGORITHM_VERSION`,
which is stored on every record and which `replay()` refuses to cross.

**And the polygon is stored anyway.** A reward signal resting on three invariants holding
across every future refactor is fragile, so every served polygon is appended to
`SERVED_STORE_PATH` and exposed at `GET /served/{task_id}` and
`GET /served/by-seed/{seed}`. This is the mitigation for open question **Q11** / divergence
**D10** — it does not *answer* Q11 (what `M_initial` means is Tier 3's decision), it
guarantees the data exists whichever way that decision goes.

---

## Telemetry transport

> **Implemented: the beacon (`TELEMETRY_TRANSPORT=beacon`).** The brief asked which of the
> two options was used and why; this is the answer, and Dev 4's `/telemetry/raw` is
> therefore a required endpoint, not a fallback.

Label Studio's stock webhook carries the final polygon and `lead_time` and nothing about
how the human got there. `C`, `L_path` and `T_dwell` have to be captured in the browser.

**Why not `annotation.meta` injection.** Two reasons, either sufficient:

1. Community Label Studio exposes no supported hook for writing `annotation.meta` before
   submit. Doing it means either forking the frontend bundle or reaching into LSF's MobX
   store, which is not a public API and changes between minor versions. Instrumentation
   bound to it breaks on upgrade, silently, in a way that looks like *"annotators produced
   no telemetry today"*.
2. The frozen `LSAnnotationUpdatedPayload` declares `effort_telemetry` at the **top level**,
   while meta injection would deliver it nested at `annotation.meta.effort_telemetry`. Dev 4
   would have to lift it before validating or every payload 422s — this is divergence
   **D5**, already flagged in the repo before either of us wrote code. The beacon matches
   the contract as frozen and the problem disappears.

**How the beacon works.** `effort_telemetry.js` hooks two stable things — DOM pointer events
on the canvas, and `fetch`/`XMLHttpRequest` calls to the annotation endpoint — rather than
LSF internals. When it sees an annotation submit starting, it fires the accumulated counters
at this service via `navigator.sendBeacon`, which survives the page teardown that follows
submit-and-advance. This service enriches the payload (filling in `wiggle_seed` from the
served-wiggle store if the browser could not read it back) and forwards it to Dev 4.

`sendBeacon` sends `text/plain`, the only JSON-carrying content type on the CORS safelist —
`application/json` would trigger a preflight, and a failed preflight makes a beacon fail
*silently*. `/telemetry/raw` reads the body raw for exactly this reason.

`TELEMETRY_TRANSPORT=ls_meta` additionally rewrites the outgoing annotation body to carry
`meta.effort_telemetry`. Whether Label Studio persists it depends on the version's
Annotation model, so it is best-effort and never the only transport. `both` runs both.

### How the three effort terms are measured

| Term | Measured as |
| --- | --- |
| `click_count` | `pointerdown` events landing on the labeling surface. Clicks on the sidebar, label picker or submit button are navigation, not correction effort, and are excluded. |
| `cursor_path_length_px` | Summed Euclidean pointer travel in **CSS pixels** — physical cursor movement, which is what "effort" means. Jumps over 400 px are ignored as window switches rather than travel. |
| `dwell_time_ms` | Time the pointer spends within `boundaryPx` (default 24) of the served polygon's outline. Each step is capped at 500 ms so a pointer parked over an edge during a coffee break does not register as hours of effort. |

CSS pixels make `L_path` depend on the annotator's zoom level. The same travel in image-pixel
space is also recorded, as `cursor_path_length_image_px` on the beacon envelope, so the
choice can be settled empirically later without re-instrumenting anything — Q16 / DD-6.

### Loading the instrumentation script

Community Label Studio has no custom-JS setting, so the script has to be injected in front
of the app. `infra/nginx/label_studio_proxy.conf` does that with `sub_filter` and serves
Label Studio on **:8081**. Label Studio's own :8080 is deliberately not published to the
host in `docker-compose.yml`.

**This is the failure mode worth guarding against:** a session on :8080 works perfectly and
produces annotations with no effort telemetry at all. Every service is green and the reward
signal is empty. Check after every deploy:

```bash
curl -s http://localhost:8081/ | grep -o 'effort_telemetry.js'   # must print a match
```

For a quick local look without the proxy, paste the script into the browser console on a
Label Studio task page, having first set
`window.RLHF_TELEMETRY_CONFIG = {endpoint: "http://localhost:8003/telemetry/raw", debug: true}`.

---

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness + config snapshot. Cheap, and deliberately does not depend on `routing_qa` — a red backend alarms annotators more than an empty queue. |
| `POST` | `/setup` | Called by Label Studio on connect. Returns `model_version`. |
| `POST` | `/predict` | The serving path. **Never returns non-200** — see below. |
| `POST` | `/webhook` | Label Studio project events. Logged only. |
| `POST` | `/telemetry/raw` | Beacon receiver. Enriches and forwards to Dev 4. |
| `GET` | `/served/{task_id}` | The polygon actually served — the action `A_t`. |
| `GET` | `/served/by-seed/{seed}` | Same, keyed by `wiggle_seed`. |
| `GET` | `/wiggle/preview` | Sample a wiggle with diagnostics, without serving it. |
| `GET` | `/static/effort_telemetry.js` | The instrumentation script. |

**`/predict` degrades instead of failing.** Label Studio treats an error from an ML backend
as a broken backend: it shows a red banner and stops calling it for the rest of the session.
An empty queue, an unreachable `routing_qa`, a contract violation, a degenerate mask, or
**image dimensions that could not be determined** all produce an empty prediction and a log
line, never a 5xx. One bad task must not end the labeling session.

That last case is deliberate and worth understanding (DD-11). If the image cannot be probed
and `IMAGE_DIM_SOURCE=probe`, the service refuses to guess the size rather than falling back
to the configured default. A wrong-scale mask does not fail visibly — the annotator corrects
it anyway, and the telemetry then describes a polygon the policy never proposed. Losing one
rollout beats corrupting the reward with a plausible-looking number. Set
`IMAGE_DIM_SOURCE=fixed` to opt into the default deliberately.

---

## Configuration

Every setting is in `.env.example` with a comment. The ones most likely to need changing:

| Variable | Default | Why you would change it |
| --- | --- | --- |
| `MOCK_MODE` | `false` | `true` to work without `routing_qa`. No automatic fallback. |
| `WIGGLE_SIGMA` | `0.02` | Calibrate with `/wiggle/preview`. |
| `WIGGLE_MODE` | `affine` | `vertex` for the 2N-dimensional alternative. |
| `IMAGE_DIM_SOURCE` | `probe` | `fixed` when images are not reachable from the container. |
| `TELEMETRY_TRANSPORT` | `beacon` | `both` to also attempt meta injection. |
| `SERVED_STORE_PATH` | `/app/data/served_wiggles.jsonl` | Must be on a volume — it is the only record of `A_t`. |

**No spec hyperparameter is set anywhere in this service.** `alpha`, `beta`, `w1..w3` and
the router thresholds are deliberately unset in the source documents and none of them are
this track's. `app/config.py` says so at the top; if one appears there, it is a bug.

---

## Known divergences

Full detail in [`docs/dev3-decisions.md`](../../docs/dev3-decisions.md). Summary of what
this track added to the ledger:

| ID | What |
| --- | --- |
| **D12** | `QueueTask` carries no image dimensions, but Label Studio polygons are percentages. Resolved inside this service by probing the image header. |
| **D14** | `WIGGLE_SIGMA` has no stated unit. Interpreted as a fraction of the object's bounding-box diagonal. |
| **D15** | The served polygon (`A_t`) is persisted, which no frozen contract requires. Mitigates Q11 / D10. |
| **D16** | Label Studio does not forward an annotator identity to an ML backend, so `annotator_id` is a placeholder unless Label Studio supplies a user in `params.context`. Dev 2's trust scores are keyed on it. |
| **D17** | Community Label Studio's webhook model has no HMAC field; the secret is sent as a header. That is a shared-secret check, not a signature — Dev 4's `LABEL_STUDIO_WEBHOOK_SECRET` verification has to match. |

Deviations from the brief's wording:

- **DD-1** — the ML Backend HTTP protocol is implemented directly on FastAPI rather than via
  the `label-studio-ml-backend` SDK. Same wire contract; no Flask alongside the FastAPI the
  rest of the stack uses.
- **DD-4** — the telemetry beacon is the implemented transport, not the fallback.
