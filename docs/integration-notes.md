# Integration notes — what Track 3 needs, and what it produces

For Dev 2 (upstream) and Dev 4 (downstream). Read the Dev 4 section before implementing
`POST /telemetry/raw`; the shape and the join rule are both here, and getting them wrong is
the predictable integration failure on this seam.

```
  Dev 1              Dev 2                    Dev 3                      Dev 4
pre_inference  →  routing_qa      →      serving_ui       →         webhook_gateway
                                            ↓  ↑
                                        Label Studio  ──────────────────────┘
                                       (annotation webhook, direct)
```

Two arrows reach Dev 4, and that is the crux of this document.

---

## For Dev 2 — what `serving_ui` calls

### `GET /tasks/next?queue=junior&annotator_id=<id>`

Called on every Label Studio `/predict`, which happens whenever an annotator opens a task.

| Expectation | Detail |
| --- | --- |
| Response body | A single `QueueTask`, validated against `common/schemas/routing_queue.py`. Validation failure is logged with field-level errors and produces an empty prediction. |
| `404` | Treated as "queue empty" — normal, not an error. The annotator gets a blank canvas. |
| `5xx` | Logged, empty prediction. `serving_ui` does not retry; Label Studio will call again on the next task open. |
| Claim semantics | The endpoint is expected to **claim** the task atomically (`pending` → `assigned`, `assigned_to` set). `serving_ui` does not claim separately, so if the endpoint is not atomic, two annotators can be served the same image — which the Tier 2 spec explicitly forbids except under IAA consensus. |
| Header | `X-Annotator-Id` is sent as well as the query parameter. |
| Timeout | `REQUEST_TIMEOUT_S`, default 5 s. |

### Fields this track actually reads

`task_id`, `image_id`, `image_url`, `bounding_box` (all four coordinates plus `label`),
`baseline_mask.points`, `queue`, `honeypot.is_honeypot`.

`routing_metrics`, `retry_count`, `status`, `assigned_to` and the timestamps are validated
but not used on the serving path.

### Two things worth knowing

**`bounding_box` is load-bearing.** Its diagonal is the length scale for the wiggle
(`WIGGLE_SCALE_REFERENCE=bbox_diagonal`), so a box that does not tightly enclose the object
makes the perturbation the wrong size. If a box is missing or degenerate, the mask's own
extent is used instead.

**`ground_truth_mask` is stripped defensively.** The plan makes "never present in any
`/tasks/next` response" a merge criterion, and Dev 2 owns that. `serving_ui` strips it again
on arrival and logs at ERROR if it was populated — belt and braces on a leak that would put
the honeypot answer key one hop from a browser. `test_api.py` asserts neither
`ground_truth_mask` nor `is_honeypot` appears in any `/predict` response.

### `POST /tasks/{task_id}/requeue`

Not called by this track. A consensus requeue results in the task being served again through
the normal path, with a **fresh** `wiggle_seed` — a re-served task explores a new action
rather than replaying one an annotator already rejected. See Q17.

---

## For Dev 4 — what `serving_ui` sends, and how to join it

### The thing that will bite you

**Effort telemetry does not arrive inside the Label Studio webhook.** Two independent
messages land at your service for one annotation:

| | Arrives from | Carries | Missing |
| --- | --- | --- | --- |
| **A. Annotation webhook** | Label Studio, directly | `annotation_id`, `result` polygon, `completed_by`, `lead_time`, timestamps | **all effort telemetry** |
| **B. Telemetry beacon** | `serving_ui`, via `POST /telemetry/raw` | `click_count`, `cursor_path_length_px`, `dwell_time_ms`, `wiggle_seed`, `task_id` | usually `annotation_id`, and the polygon |

`LSAnnotationUpdatedPayload` describes the *merged* record. Neither message satisfies it
alone. If you validate message A against that model, `effort_telemetry` will be missing and
every webhook 422s.

**Why it is this way:** stock Label Studio has no supported hook for attaching custom data to
an annotation before submit, and the frozen contract puts `effort_telemetry` at the top level
rather than nested under `meta` — which is divergence **D5**, already in the repo's ledger.
Full reasoning in [`dev3-decisions.md`](dev3-decisions.md) DD-4.

### Update 2026-08-19 — `annotation_id` is now sent, and please do not synthesise one

**What changed:** the beacon now reads `annotation_id` off the **response** to the annotation
submit, so in the normal case it carries the real id and joins to the webhook exactly. It is
still `null` when the page tears down before the response arrives — a safety net, because
losing the id is recoverable while losing the whole effort record is not.

**Please do not generate a surrogate id.** A synthesised `raw_{task_id}_{session_id}` used as
`idempotency_key` does keep the endpoint from erroring, but it breaks the thing the endpoint
exists for:

* the beacon envelope gets the fake key, the real `ANNOTATION_UPDATED` webhook gets the real
  `annotation_id`;
* they are then two unrelated envelopes on `telemetry:ingest` that nothing can pair up;
* Tier 3 needs **both halves of one record** — `ΔIoU` comes from the final polygon, which
  only the webhook carries, and `ΔE` comes from the effort terms, which only the beacon
  carries. `R_t = α·ΔIoU − β·ΔE_norm` cannot be computed from either half alone.

Nothing crashes. The queue fills with half-records, and the failure only becomes visible when
Tier 3 tries to compute a reward and finds no matching pair.

**Join on `wiggle_seed` instead.** It is on both messages, it is unique per served task, and
it does not depend on Label Studio's id timing. `task_id` is the fallback.

If a surrogate is genuinely needed for dedupe within your own store, keep it in a separate
field rather than in `annotation_id` / `idempotency_key`, so the real id stays available for
the join.

### The join rule

In order of preference:

1. **`wiggle_seed`** — the strongest key. Minted per served task by `serving_ui`, unique, and
   present on both messages: on the beacon directly, and on the webhook inside the
   prediction's `model_version` (`serving-ui-stochastic-0.1.0|seed=<hex>`) and in the
   region's `meta.text` (`wiggle_seed=<hex>`).
2. **`task_id`** — our `QueueTask.task_id`, carried in Label Studio's `task.data.task_id` and
   on the beacon. Stable, but **reused across a consensus requeue**, so pair it with the
   newest unmatched beacon rather than assuming one-to-one.
3. **`client_session_id`** — per browser tab. Disambiguates two annotators open on the same
   task.

`annotation_id` is present whenever the submit response comes back before the page
navigates, which is the normal case — the instrumentation reads it off that response. It is
`null` only when the tab tears down first. When present, the join is exact; when null, use
`wiggle_seed`.

### Suggested handling

Because the two messages race, neither should block on the other:

- Buffer beacons in Redis keyed by `wiggle_seed` (and `task_id`) with a TTL of a few minutes.
- On an annotation webhook, look up the buffered beacon, merge `effort_telemetry` into the
  payload, then build the `RedisEventEnvelope` as specified.
- **If no beacon is found, still enqueue the envelope**, flagged. An annotation with no
  telemetry is a valid annotation that produces no rollout — dropping it loses the human's
  work as well as the signal. It is also the symptom of the instrumentation script not being
  loaded (see below), so it needs to be visible in a metric rather than silent.

`idempotency_key = annotation_id` still works: it lives on message A, which is the one that
triggers the enqueue.

### The `/telemetry/raw` payload

Exact shape in `app/models.py::RawTelemetryEnvelope`; a real capture is in
[`tests/mocks/telemetry_raw_payload.json`](../tests/mocks/telemetry_raw_payload.json).

```json
{
  "task_id": "task_a1b2c3",
  "effort_telemetry": {
    "click_count": 17,
    "cursor_path_length_px": 1284.6,
    "dwell_time_ms": 6820,
    "wiggle_seed": "c644b8cf379b545910d076f8e05d913c"
  },
  "annotation_id": null,
  "project_id": "1",
  "completed_by": "annotator_42",
  "ls_task_id": 7,
  "client_session_id": "sess_k3n8fq2xa1",
  "wiggle_seed": "c644b8cf379b545910d076f8e05d913c",
  "cursor_path_length_image_px": 1284.6,
  "lead_time": 41.3,
  "client_sent_at": "2026-08-18T10:16:39.412Z",
  "transport": "beacon"
}
```

- `effort_telemetry` validates against the frozen `LSTelemetryMeta` unchanged — so you can
  reuse the model directly.
- `wiggle_seed` is duplicated at the top level purely for join convenience; the two are
  always the same value.
- `cursor_path_length_image_px` is diagnostic (see Q16). Ignore it if you like.
- **This shape is not yet a frozen contract** — it is proposed additively as **Q15**. If you
  would rather it lived in `common/schemas/`, say so and it becomes a schema PR.

`serving_ui` enriches before forwarding: a missing `wiggle_seed` is recovered from the
served-wiggle store, and `completed_by` / `project_id` are filled in from the serve record
when the browser did not supply them.

### Webhook authentication — read this before implementing HMAC

The plan says *"validate HMAC signature (`LABEL_STUDIO_WEBHOOK_SECRET`)"*. **Community Label
Studio's webhook model has no HMAC-signing field.** It sends arbitrary custom headers.

`setup_project.py` therefore registers the secret as:

```
X-Label-Studio-Secret: <LABEL_STUDIO_WEBHOOK_SECRET>
```

That is a shared-secret check, not a signature — it proves possession of the secret but does
not bind it to the request body. Divergence **D17**. If you implement true HMAC verification,
webhooks will fail with a 401 and the cause will not be obvious from either side.

### Recovering the action `A_t`

Tier 3 computes `ΔIoU` against the mask the human actually corrected — the **wiggled** one,
which is the sampled action, not `baseline_mask`. That is open question **Q11** / divergence
**D10**, the repo's most consequential contract gap.

`serving_ui` persists the served polygon and exposes it:

```
GET http://serving_ui:8003/served/{task_id}
GET http://serving_ui:8003/served/by-seed/{wiggle_seed}
```

Returns `baseline_points` (μ, canonicalised), `wiggled_points` (`A_t`, absolute pixels),
the full `wiggle_params`, the 5-dimensional `action` vector, and the image dimensions.
Sample: [`tests/mocks/served_wiggle_record.json`](../tests/mocks/served_wiggle_record.json).

**Coordinate spaces do not match and this is easy to get wrong.** `result` regions on the
webhook are Label Studio **percentages, 0–100**; `wiggled_points` are **absolute pixels**.
Convert before computing any IoU — `ls_format.from_percent(points, width, height)` exists
for exactly this, and the record carries the `image_width` / `image_height` to use.

---

## Verifying the seam without the other services

Everything below runs with only `serving_ui` up.

```bash
# 1. Bring up Track 3 in fixture mode
MOCK_MODE=true IMAGE_DIM_SOURCE=fixed \
DEFAULT_IMAGE_WIDTH=800 DEFAULT_IMAGE_HEIGHT=600 \
python -m uvicorn app.main:app --app-dir services/serving_ui --port 8003

# 2. Ask for a prediction exactly as Label Studio would
curl -s -X POST localhost:8003/predict -H 'Content-Type: application/json' \
     -d @tests/mocks/ls_predict_request.json | python -m json.tool

# 3. Pull the seed out and fetch the action A_t
SEED=$(curl -s -X POST localhost:8003/predict -H 'Content-Type: application/json' \
       -d @tests/mocks/ls_predict_request.json \
       | python -c "import sys,json;print(json.load(sys.stdin)['results'][0]['model_version'].split('|seed=')[1])")
curl -s "localhost:8003/served/by-seed/$SEED" | python -m json.tool

# 4. Fire a telemetry beacon the way the browser does
curl -s -X POST localhost:8003/telemetry/raw \
     -H 'Content-Type: text/plain;charset=UTF-8' \
     -d @tests/mocks/telemetry_raw_payload.json | python -m json.tool
```

With `webhook_gateway` running and `TELEMETRY_FORWARD_ENABLED=true`, step 4 forwards to
`WEBHOOK_GATEWAY_URL/telemetry/raw` and the response reports `"forwarded": true`.

---

## Pre-merge checklist for this seam

- [ ] Dev 4's `/telemetry/raw` accepts `RawTelemetryEnvelope`, including a `null`
      `annotation_id` and a `text/plain` content type.
- [ ] Dev 4's `/webhooks/label-studio` does **not** require `effort_telemetry` on the inbound
      payload, and does not 422 without it.
- [ ] Dev 4's secret check is a header comparison, not an HMAC of the body (D17).
- [ ] Dev 2's `/tasks/next` claims atomically and never returns `ground_truth_mask`.
- [ ] `curl -s http://localhost:8081/ | grep effort_telemetry.js` prints a match. **Without
      this, every annotation collected is telemetry-free and the reward signal is empty
      while every service reports healthy.**
- [ ] An end-to-end annotation produces both a beacon and a webhook that join on
      `wiggle_seed`.
