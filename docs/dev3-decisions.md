# Developer 3 — decisions, deviations, and new open questions

Every judgment call this track made that a reviewer could reasonably have made differently,
with the reasoning attached. Written so the next agent — which may not be Claude — can
disagree with a decision on its merits rather than having to reverse-engineer it.

Three kinds of entry:

- **DD-n** — a decision this track made, including deviations from the brief's wording.
- **D-n** — a spec-vs-code divergence, numbered to continue
  [`docs/implementation/spec-alignment.md`](../../docs/implementation/spec-alignment.md),
  which ends at D11.
- **Q-n** — a question for a human, numbered to continue
  [`docs/reference/open-questions.md`](../../docs/reference/open-questions.md), which ends
  at Q13.

The D and Q entries below should be folded into those two ledgers when this track merges.
They are kept here for now so the track repo is self-contained.

---

## Decisions

### DD-1 — The ML Backend protocol is implemented on FastAPI, not the SDK

**The brief says** *"use the label-studio-ml-backend SDK"*.

**What was built:** the ML Backend HTTP contract implemented directly with FastAPI —
`GET /health`, `POST /setup`, `POST /predict`, `POST /webhook`.

**Why.** The SDK wraps four JSON endpoints in a Flask application. Taking it would put Flask
in a service that sits alongside three FastAPI services, pull its transitive pins into a
`common/schemas` shared with three other tracks, and constrain the extra endpoints this
track needs (`/telemetry/raw`, `/served/*`, `/wiggle/preview`) to live inside its app object.
The SDK also churns across versions in ways that have historically broken `predict()`
signatures.

**What is unchanged:** the wire contract. Label Studio cannot tell the difference, which is
the point — `tests/mocks/ls_predict_request.json` and `ls_predict_response.json` are real
captures of it.

**If you disagree:** the swap is contained to `app/main.py`. Everything the SDK would call —
`wiggle.py`, `ls_format.py`, `task_source.py` — is already independent of the web framework.

---

### DD-2 — The wiggle is a 5-dimensional affine action, not per-vertex noise

**The brief says** *"perturb each vertex independently vs a shared affine jitter, pick one"*.

**Chosen:** shared affine, 5 dimensions — `tx`, `ty`, `log_scale`, `theta`, `normal_offset`.

**Why.** Tier 2 §2 is explicit that the action space is *"low-dimensional latent points and
boxes, never a raw pixel grid"*, because the critic must attribute reward to a handful of
parameters rather than to a million pixels (**Fault 2**). Per-vertex noise on a 200-vertex
polygon is a 400-dimensional action — the same credit-assignment problem in a different
coordinate system. It also *looks* different to the annotator: jagged and broken rather than
a near-miss, which pushes people to delete and redraw. A redraw produces effort telemetry
that describes starting over, not correcting, and those are not the same signal.

`WIGGLE_MODE=vertex` implements the alternative so the two can be compared empirically.

---

### DD-3 — The action includes a boundary-normal offset

Translation, rotation and scale can only *relocate* a shape. The error annotators actually
correct is a boundary that is slightly too fat or too thin, which none of those three can
produce. `normal_offset` pushes every vertex along its outward normal, so the sampled masks
span the error mode the humans are actually there to fix.

It is also the dimension that most resembles Tier 1's **Deliberate Degradation Engine**
(a ~20 px polygon expansion). At `sigma=0.02` on the shared fixture the mean displacement
comes out at **≈21 px** — close enough that Q10 (*does the wiggle replace the degradation
engine?*) deserves re-asking with this number in hand. It is not an answer: the degradation
engine applies to a *subset* of already-perfect masks as a QA gate, while the wiggle applies
to *every* task for exploration. Same magnitude, different job. See **Q14** below.

---

### DD-4 — The telemetry beacon is the implemented transport

**The brief offers** meta injection into `annotation.meta.effort_telemetry`, with a
`POST /telemetry/raw` beacon as the fallback, and asks which was used.

**Implemented: the beacon.** Meta injection ships as a best-effort secondary layer
(`TELEMETRY_TRANSPORT=ls_meta` or `both`) and is never relied on alone.

**Why**, in the order that matters:

1. **The frozen contract already prefers it.** `LSAnnotationUpdatedPayload` declares
   `effort_telemetry` at the **top level**. Meta injection delivers it nested, so Dev 4
   would have to lift the block before validating or reject every payload. That is
   divergence **D5**, flagged in this repo before either of us wrote code. The beacon
   matches the contract as frozen.
2. **Community Label Studio has no supported hook.** Writing `annotation.meta` before submit
   means forking the frontend bundle or reaching into LSF's MobX store, which is not a
   public API and changes between minor versions. Instrumentation bound to it fails silently
   on upgrade — annotations keep flowing, telemetry quietly stops.
3. **Beacons survive teardown.** `navigator.sendBeacon` completes after the page navigates.
   A `fetch` can be cancelled when LSF advances to the next task, which would lose telemetry
   preferentially for the *fastest* completions — a systematic bias in exactly the variable
   being measured.

**Cost of this choice:** telemetry and the annotation arrive at Dev 4 separately and must be
joined. The join rule is specified in [`integration-notes.md`](integration-notes.md).

---

### DD-5 — Failures on `/predict` degrade to an empty prediction

Label Studio treats a non-200 from an ML backend as a broken backend: red banner, and it
stops calling it for the rest of the session. So an empty queue, an unreachable
`routing_qa`, a contract violation, or a degenerate mask all produce an empty prediction and
a log line rather than an error status.

**The trade:** a silent empty canvas is harder to notice than a banner. Mitigated by logging
every case at ERROR with the task id, and by `/health` reporting the task source. A task
served with no pre-annotation still yields a usable annotation — it just yields no *rollout*,
because there was no sampled action to attribute the correction to.

---

### DD-6 — `cursor_path_length_px` is CSS pixels; image pixels are recorded alongside

`L_path` is *"summed Euclidean cursor travel"*, and the spec does not say in which space.

- **CSS pixels** measure physical cursor movement — what "effort" means — but make the value
  depend on the annotator's zoom level.
- **Image pixels** are zoom-invariant and comparable across annotators, but a zoomed-in
  annotator who moved their hand twice as far records the same number.

The frozen field carries **CSS pixels**, since E-DRDE is a model of human effort. The
image-space equivalent rides along as `cursor_path_length_image_px` on the beacon envelope,
so the choice can be revisited from real data without re-instrumenting the frontend. See
**Q16**.

---

### DD-7 — Dwell is capped per sample; clicks are filtered to the canvas

`T_dwell` accumulates in steps capped at 500 ms, and stops when the tab loses focus. Without
a cap, a pointer left over a boundary during a phone call registers as hours of effort and
that one rollout dominates its batch. Dwell is meant to measure hesitation over a hard edge.

`click_count` only counts `pointerdown` on the labeling surface. Clicks on the sidebar,
label picker and submit button are navigation, and counting them would make `C` partly a
measure of UI chrome.

Both are heuristics with thresholds this track picked. They are named constants in the JS,
not spec values.

---

### DD-8 — The seed rides three channels

`wiggle_seed` has to reach the browser and come back. It is carried in the region's
`meta.text`, appended to `model_version`, and retrievable from `GET /served/{task_id}`;
the server also fills a missing seed in during enrichment.

Redundant on purpose. Losing the seed does not break the annotation — it breaks the link
between the effort and the action that caused it, which is the entire reward signal, and it
does so invisibly.

---

### DD-9 — Instrumentation is delivered by a reverse proxy

Community Label Studio has no custom-JS setting. The options were: fork the frontend bundle,
ask every annotator to install a userscript, or inject a `<script>` tag in front of the app.
Only the third is a deployment concern rather than a per-annotator chore, so
`infra/nginx/label_studio_proxy.conf` injects it and serves Label Studio on **:8081**.
Label Studio's own :8080 is not published to the host.

**The failure mode this creates:** a session on :8080 works perfectly and captures no
telemetry. Everything is green; the reward signal is empty. Hence the injection check in the
README and in the proxy config.

---

### DD-11 — A failed dimension probe suppresses the pre-annotation

Originally a failed probe fell back to `DEFAULT_IMAGE_WIDTH`/`HEIGHT` with a warning. Found
in practice to be dangerous: probing `https://images.cocodataset.org/val2017/...` fails
certificate verification (the host's cert does not match — reproduced in both `httpx` and
`curl`, so it is the host, not the machine), and a **640×480 image silently became
1920×1080**. A 3× scale error puts the polygon far off-target.

That failure is worse than it looks. The annotator corrects the misplaced mask anyway,
because correcting masks is the job — and the resulting effort telemetry then describes a
polygon the policy never proposed. The reward signal is not *missing*, it is *wrong*, and it
looks entirely plausible downstream.

So `ImageDims` now carries a `reliable` flag, and `/predict` returns an empty prediction when
it is false. Losing one rollout is the cheaper failure.

`IMAGE_DIM_SOURCE=fixed` stays reliable by definition — that is an operator explicitly
choosing the number, which is different from nobody choosing it.

### DD-12 — This service serves local test images itself

`serving_ui` exposes `ASSETS_DIR` (default `tests/assets/`) at `/assets/`, and reads image
URLs under that prefix **straight off disk** rather than fetching its own socket.

Label Studio's own local-file serving was the obvious alternative and is worse here: it needs
a document root, a volume, and an authenticated `/data/local-files/?d=…` URL that the
dimension probe cannot fetch — so the browser would load the image while the probe failed,
which is exactly the DD-11 trap. Serving it from here means the browser and the probe read
the same plain-HTTP URL.

Only the assets directory is exposed, and traversal outside it is refused. `tests/` itself is
deliberately **not** mounted or copied wholesale, in the Dockerfile or in compose: it holds
the hand-dropped Label Studio API token during local setup.

### DD-10 — The frozen contracts were not edited

`common/schemas/label_studio_webhook.py` is Dev 3's to own, and there were two temptations to
edit it: adding a nested-vs-top-level note for `effort_telemetry` (D5), and adding fields the
beacon needs. Neither was done. The plan requires a schema change to be its own PR reviewed
by the downstream consumer, and Dev 4 validates against this file.

Instead, `app/models.py` holds `RawTelemetryEnvelope` as a **proposed additive** contract,
and `scripts/check_schema_drift.py` fails if this repo's vendored copy diverges from the
scaffold's — so the D7 duplication mistake cannot repeat silently.

---

## New divergences (continuing the D-series)

### D12 — `QueueTask` carries no image dimensions · **MISMATCH**

`PolygonMask.points` are absolute pixels (`tier1_ingestion.py` says so). Label Studio stores
polygon coordinates as **percentages of the image, 0–100** — visible in Dev 4's own fixture,
whose points are values like `[10.1, 12.4]`. Converting needs width and height, and **no
frozen contract carries them.** `QueueTask` has `image_url` and a pixel-space `bounding_box`.

**Resolved inside this service** rather than by a schema change: `app/image_meta.py` reads
the dimensions from the image header (PNG/JPEG/GIF/BMP/WEBP, hand-parsed, no Pillow),
preferring `width`/`height` on the Label Studio task data when present, and falling back to
`DEFAULT_IMAGE_WIDTH`/`HEIGHT` with a loud warning.

**Blast radius if it goes wrong:** wrong dimensions do not fail — they render the mask in the
wrong place, and the annotator silently corrects a polygon that was never the model's output.
The telemetry looks valid and the reward is garbage. Worth a `width`/`height` field on
`GroundedSAM2Output` if the contracts are ever reopened.

### D14 — `WIGGLE_SIGMA` has no stated unit · **NEW**

`.env.example` sets `WIGGLE_SIGMA=0.02` and no document says what it is a fraction of. This
track interprets it as a fraction of the object's **bounding-box diagonal**
(`WIGGLE_SCALE_REFERENCE=bbox_diagonal`), falling back to the mask's own extent. On the
shared fixture that gives ≈21 px mean displacement and ≈0.83 IoU against the baseline.

The value 0.02 is not invented — it is the spec's. Its *interpretation* is this track's, and
it is a real choice: read as a fraction of image width instead, the same number would behave
completely differently on a small object. See **Q14**.

### D15 — The served polygon is persisted, which no contract requires · **NEW**

`app/store.py` appends every served wiggle to a JSONL store and exposes it at `/served/*`.

Nothing in the MVP consumes it. It exists because **Q11 / D10** — the repo's own
most-consequential open question — turns on whether `A_t` can be recovered, and `wiggle_seed`
only makes it *reconstructible*, contingent on the RNG, vertex ordering and transform all
staying pinned forever. One JSONL append per served task turns that into a lookup.

This does **not** answer Q11. It guarantees the data exists whichever way Q11 is decided.

### D16 — `annotator_id` is a placeholder in practice · **MISMATCH**

Label Studio does not forward an annotator identity to an ML backend. `_resolve_annotator`
tries the `X-Annotator-Id` header, then `params.context.user`, then `params.login`, then
falls back to `DEFAULT_ANNOTATOR_ID`.

**Consequence:** Dev 2's honeypot trust scores are keyed on `annotator_id`, and every
annotator looks like the same person when the fallback is used — the Trust Score Engine
cannot distinguish them, so a failed honeypot penalises everyone or no one. The plan already
defers real auth (§0, "MVP header stub"), so this is a known consequence of a known deferral,
recorded so it is not discovered later as a bug in trust scoring.

### D17 — The webhook secret is a header, not an HMAC signature · **MISMATCH**

The plan has Dev 4 *"validate HMAC signature (`LABEL_STUDIO_WEBHOOK_SECRET`)"*. Community
Label Studio's webhook model has **no HMAC-signing field** — it sends arbitrary custom
*headers*. `setup_project.py` therefore registers the secret as `X-Label-Studio-Secret`.

That is a shared-secret check, not a signature: it proves possession of the secret but does
not bind it to the body, so it does not detect tampering in transit. Acceptable for an MVP
on a private network; **Dev 4's verification has to match this or every webhook 401s.**

---

## New open questions (continuing the Q-series)

### Q14 — Is `WIGGLE_SIGMA` a fraction of the object, and does 0.02 do the right job?

Two halves.

**Unit.** Implemented as a fraction of the bounding-box diagonal (D14). Alternatives: a
fraction of image width, or absolute pixels. The choice changes behaviour most on very small
and very large objects.

**Magnitude, and its relationship to Q10.** At `sigma=0.02` the mean displacement is ≈21 px,
which is close to the Deliberate Degradation Engine's ~20 px expansion. Q10 asks whether the
wiggle replaces the degradation engine; that was posed on the assumption the wiggle would be
*far subtler*, which at this scale reference it is not. The two still differ in *scope* —
degradation targets a subset of perfect masks as a QA gate, the wiggle applies to every task
for exploration — so this is not an answer, but Q10 should be re-asked with the measured
number rather than the assumed one.

`/wiggle/preview` exists to make this an empirical question: it reports IoU against the
baseline and mean displacement for any sigma.

### Q15 — Should `RawTelemetryEnvelope` become a frozen contract?

The beacon needs to carry identifiers (`task_id`, `wiggle_seed`, `client_session_id`) that
`LSTelemetryMeta` does not have, because effort telemetry alone cannot be joined to an
annotation. It currently lives in `app/models.py` as a proposed additive contract rather
than in `common/schemas/`, since the freeze rule says a schema change is its own reviewed PR.

Dev 4 has to implement `/telemetry/raw` against *something*. Either this shape gets promoted
into `common/schemas/`, or Dev 4 defines their own and the two are reconciled at integration
— which is the more expensive order. Shape and rationale in
[`integration-notes.md`](integration-notes.md).

### Q16 — Is `L_path` physical travel or image-space travel?

CSS pixels measure what the human's hand did but vary with zoom; image pixels are
zoom-invariant but under-count a zoomed-in annotator's real effort. Both are recorded
(DD-6). Settling it needs real sessions, and it interacts with **Q3** — if `w2` turns out to
be near zero, the question is moot.

### Q17 — Should a re-served task explore somewhere new?

Each serve mints a fresh seed, so a task requeued after a consensus retry gets a genuinely
new action rather than replaying one an annotator already rejected. That seemed obviously
right — resampling is what exploration means — but it means the same task can contribute
several rollouts with correlated baselines to the same PPO batch, which interacts with
**Q1**'s batch-diversity concern. Flagged rather than resolved.
