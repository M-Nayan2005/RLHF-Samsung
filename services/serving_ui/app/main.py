"""
serving_ui - the Label Studio ML Backend for Tier 2.

This is the service Label Studio calls when an annotator opens a task. It claims
the next task from Dev 2's router, samples an action from the stochastic policy
(`wiggle.py`), and returns the deliberately-slightly-wrong polygon as a Label
Studio pre-annotation. The human corrects it; the effort that correction costs is
the reward signal the rest of the system is built on.

Endpoints
---------
Label Studio ML Backend protocol:
    GET  /health              liveness, config snapshot, upstream identity
    POST /setup               called when the backend is connected to a project
    POST /predict             the main path: claim -> wiggle -> pre-annotate
    POST /webhook             Label Studio project events (accepted, not required)

This track's own:
    POST /telemetry/raw       beacon receiver, enriches and forwards to Dev 4
    GET  /served/{task_id}    the polygon actually served (the action A_t)
    GET  /served/by-seed/{s}  same, keyed by wiggle_seed
    GET  /wiggle/preview      sample a wiggle without serving it, for calibration
    GET  /static/...          serves the instrumentation script to the browser

Deviation from the brief: the brief says to use the `label-studio-ml-backend`
SDK. This implements the ML Backend HTTP protocol directly in FastAPI instead -
reasoning in `docs/dev3-decisions.md` (DD-1), and the wire contract is unchanged
either way.
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from common.schemas.routing_queue import QueueTask

from . import ls_format
from .config import Settings, get_settings, startup_warnings
from .image_meta import ImageDimensionResolver, ImageDims
from .models import (
    LSPredictRequest,
    LSPredictResponse,
    LSSetupRequest,
    LSTask,
    RawTelemetryEnvelope,
    ServedWiggleRecord,
    TelemetryAck,
    WigglePreview,
)
from .store import ServedWiggleStore
from .task_source import NoTaskAvailable, TaskSourceError, build_task_source
from .telemetry import TelemetryForwarder, utc_now_iso
from .wiggle import WIGGLE_ALGORITHM_VERSION, WiggleError, apply_wiggle, new_seed

log = logging.getLogger("serving_ui")

# The instrumentation script is served from where it lives, rather than copied
# into a build artefact. One file on disk means the version the browser runs is
# unambiguously the version in the repo.
STATIC_DIR = Path(__file__).resolve().parent.parent / "label_studio" / "instrumentation"


def configure_logging(settings: Settings) -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level, logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    configure_logging(settings)

    log.info("serving_ui starting on port %s (env=%s)", settings.port, settings.environment)
    log.info("task source: %s", app.state.task_source.describe())
    log.info("telemetry transport: %s -> %s", settings.telemetry_transport, app.state.forwarder.endpoint)
    log.info("served-wiggle store: %s", app.state.store.stats())
    log.info(
        "policy: mode=%s sigma=%s scale_reference=%s algorithm=%s",
        settings.wiggle_mode, settings.wiggle_sigma,
        settings.wiggle_scale_reference, WIGGLE_ALGORITHM_VERSION,
    )
    for warning in startup_warnings(settings):
        log.warning(warning)

    yield

    for client in (app.state.http_client, app.state.probe_client):
        client.close()


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="serving_ui - Tier 2 Interactive Serving",
        description=__doc__,
        version="0.1.0",
        lifespan=lifespan,
    )

    http_client = httpx.Client(timeout=settings.request_timeout_s)
    probe_client = httpx.Client(timeout=settings.image_probe_timeout_s, follow_redirects=True)
    store = ServedWiggleStore(settings)

    app.state.settings = settings
    app.state.http_client = http_client
    app.state.probe_client = probe_client
    app.state.store = store
    app.state.task_source = build_task_source(settings, client=http_client)
    app.state.dimensions = ImageDimensionResolver(settings, client=probe_client)
    app.state.forwarder = TelemetryForwarder(settings, store, client=http_client)

    # The instrumentation script runs on the Label Studio origin (:8080) and posts
    # telemetry here (:8003), so the beacon is cross-origin. Without this, the
    # browser drops it before it is ever sent.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(settings),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Local test images, served over plain HTTP from this service.
    #
    # Label Studio's own local-file serving needs a document root, a volume and
    # an authenticated /data/local-files/ URL that the dimension probe cannot
    # fetch. Serving the image from here instead means the browser and the probe
    # read the same URL, with no TLS and no auth in the way - which matters,
    # because the first "reachable" image tried (images.cocodataset.org) failed
    # certificate verification in both httpx and curl.
    #
    # Only `tests/assets/` is exposed. `tests/` itself is NOT mounted: it holds
    # the hand-dropped Label Studio API token during local setup.
    assets_dir = Path(settings.assets_dir)
    if assets_dir.exists():
        app.mount(
            settings.assets_url_prefix.rstrip("/") or "/assets",
            StaticFiles(directory=str(assets_dir)),
            name="assets",
        )

    _register_routes(app)
    return app


def _cors_origins(settings: Settings) -> List[str]:
    """
    Origins allowed to post telemetry.

    Defaults to the configured Label Studio URL plus its localhost spelling,
    since a browser reaching Label Studio at http://localhost:8080 sends that as
    the Origin even when the container knows itself as http://label_studio:8080.
    `CORS_ALLOW_ORIGINS=*` is available for local debugging and is refused a
    mention in the compose file on purpose.
    """
    import os

    configured = os.environ.get("CORS_ALLOW_ORIGINS", "").strip()
    if configured:
        return [o.strip() for o in configured.split(",") if o.strip()]

    origins = {settings.label_studio_url}
    port = settings.label_studio_url.rsplit(":", 1)[-1]
    if port.isdigit():
        origins.add(f"http://localhost:{port}")
        origins.add(f"http://127.0.0.1:{port}")
    return sorted(origins)


def _register_routes(app: FastAPI) -> None:  # noqa: C901 - a flat route table reads better than five modules

    # ------------------------------------------------------------------
    # ML Backend protocol
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> Dict[str, Any]:
        """
        Label Studio polls this to decide whether the backend is connected, so it
        must stay cheap and must not depend on routing_qa being up - a red
        backend in the Label Studio UI is far more alarming to an annotator than
        an empty queue.
        """
        settings: Settings = app.state.settings
        return {
            "status": "UP",
            "model_version": settings.model_version,
            "task_source": app.state.task_source.describe(),
            "store": app.state.store.stats(),
            "config": settings.redacted(),
        }

    @app.post("/setup")
    async def setup(request: Request) -> Dict[str, Any]:
        """
        Called when the backend is attached to a project or the config is saved.

        Reads the body raw and **never fails validation**. Label Studio treats any
        non-200 from /setup as "this doesn't look like a valid ML backend" and
        refuses to connect at all - so a strict model here turns a cosmetic
        payload difference into a total integration failure, with an error
        message that blames the model rather than the parser.

        That is not hypothetical: Label Studio 1.23.0 sends `project` as a value
        this service's model typed as `str`, and Pydantic v2 does not coerce it.
        The connection failed with "it might be incompatible with the current
        labeling configuration", which points at entirely the wrong thing.

        The only part of this payload that matters is the labeling config, and
        even that is re-read on every /predict call.
        """
        settings: Settings = app.state.settings

        raw: Dict[str, Any] = {}
        try:
            parsed = await request.json()
            if isinstance(parsed, dict):
                raw = parsed
        except Exception as exc:  # noqa: BLE001 - any parse failure is non-fatal here
            log.warning("setup body was not JSON (%s); continuing anyway", exc)

        label_config = raw.get("label_config") or raw.get("schema")
        if not isinstance(label_config, str):
            label_config = None

        info = ls_format.parse_label_config(label_config, settings)
        log.info(
            "setup: project=%r keys=%s -> from_name=%s to_name=%s labels=%s",
            raw.get("project"), sorted(raw), info.from_name, info.to_name,
            info.labels or "<none declared>",
        )
        return {"model_version": settings.model_version}

    @app.post("/predict", response_model=LSPredictResponse)
    def predict(
        payload: LSPredictRequest,
        annotator_header: Optional[str] = Header(default=None, alias="X-Annotator-Id"),
    ) -> LSPredictResponse:
        """
        The serving path.

        For each task Label Studio asks about: claim a `QueueTask` from the
        router, sample a wiggle, persist what was served, and return the
        pre-annotation.

        Failures degrade to an empty prediction rather than an HTTP error. Label
        Studio surfaces a non-200 from an ML backend as a broken-backend banner
        and stops calling it for the rest of the session; an empty prediction
        just means the annotator draws from scratch. One task's problem must not
        take the labeling session down.
        """
        settings: Settings = app.state.settings
        info = ls_format.parse_label_config(payload.label_config, settings)
        annotator_id = _resolve_annotator(payload, annotator_header, settings)

        results = []
        for ls_task in payload.tasks or [LSTask()]:
            results.append(
                _predict_one(
                    app, ls_task, info, annotator_id,
                    ls_project_id=payload.project, settings=settings,
                )
            )

        return LSPredictResponse(results=results, model_version=settings.model_version)

    @app.post("/webhook")
    def ls_webhook(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, str]:
        """
        Label Studio project-event webhook.

        Accepted and logged only. The annotation webhook that matters goes
        straight to Dev 4's gateway - routing it through here first would add a
        hop to the path the plan requires to answer in under 50ms.
        """
        log.info("Label Studio project event: action=%s", payload.get("action"))
        return {"status": "ok"}

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    @app.post("/telemetry/raw", response_model=TelemetryAck)
    async def telemetry_raw(request: Request) -> TelemetryAck:
        """
        Beacon receiver: the browser posts effort telemetry here just before submit.

        The body is read raw rather than declared as a Pydantic parameter because
        `navigator.sendBeacon` sends `text/plain` — the only content type that
        avoids a CORS preflight, and a preflight that fails makes a beacon fail
        *silently*, which would drop telemetry with nothing in any log to say so.

        Always 200s on a well-formed body, even when the forward to Dev 4 fails.
        The browser fires this during page teardown and cannot act on an error,
        and the annotator's submit must not be coupled to the gateway's health.
        """
        raw = await request.body()
        try:
            envelope = RawTelemetryEnvelope.model_validate_json(raw)
        except ValueError as exc:
            log.warning("Rejected malformed telemetry beacon: %s | body=%.500s", exc, raw)
            raise HTTPException(status_code=422, detail=f"malformed telemetry: {exc}") from exc

        forwarder: TelemetryForwarder = app.state.forwarder
        enriched = await run_in_threadpool(forwarder.enrich, envelope)
        forwarded, detail = await run_in_threadpool(forwarder.forward, enriched)
        return TelemetryAck(
            accepted=True,
            task_id=enriched.task_id,
            forwarded=forwarded,
            detail=detail,
        )

    # ------------------------------------------------------------------
    # Served-wiggle lookups (Q11 / D10 - see store.py)
    # ------------------------------------------------------------------

    @app.get("/served/{task_id}", response_model=ServedWiggleRecord)
    def served_by_task(task_id: str) -> ServedWiggleRecord:
        record = app.state.store.by_task(task_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no served wiggle recorded for task {task_id}")
        return record

    @app.get("/served/by-seed/{wiggle_seed}", response_model=ServedWiggleRecord)
    def served_by_seed(wiggle_seed: str) -> ServedWiggleRecord:
        record = app.state.store.by_seed(wiggle_seed)
        if record is None:
            raise HTTPException(status_code=404, detail=f"no served wiggle recorded for seed {wiggle_seed}")
        return record

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    @app.get("/wiggle/preview", response_model=WigglePreview)
    def wiggle_preview(
        sigma: Optional[float] = Query(None, ge=0, description="Overrides WIGGLE_SIGMA for this call"),
        mode: Optional[str] = Query(None, description="affine | vertex"),
        seed: Optional[str] = Query(None, description="Reuse a seed to reproduce a specific wiggle"),
        queue: Optional[str] = Query(None),
        annotator_id: str = Query("preview"),
    ) -> WigglePreview:
        """
        Sample a wiggle and return it with diagnostics, without serving it.

        This is how sigma gets calibrated by looking rather than guessing:
        `iou_vs_baseline` near 1.0 means the annotator has nothing to correct and
        the rollout carries no signal; near 0 means they will redraw from scratch
        and the correction no longer relates to the sampled action. The result is
        deliberately NOT written to the served-wiggle store - a preview is not a
        serve, and recording it would put a polygon no human ever saw into the
        data Tier 3 reads.
        """
        settings: Settings = app.state.settings
        task = _claim_task(app, queue or settings.junior_queue_name, annotator_id)
        dims = app.state.dimensions.resolve(task.image_url)

        try:
            result = apply_wiggle(
                task.baseline_mask.points,
                seed=seed or new_seed(settings.wiggle_seed_bytes),
                sigma=settings.wiggle_sigma if sigma is None else sigma,
                mode=mode or settings.wiggle_mode,
                scale_reference=settings.wiggle_scale_reference,
                bounding_box=_box_of(task),
                image_width=dims.width,
                image_height=dims.height,
                clamp=settings.wiggle_clamp_to_frame,
            )
        except WiggleError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return WigglePreview(
            baseline_points=result.baseline_points,
            wiggled_points=result.points,
            ls_points_percent=ls_format.to_percent(result.points, dims.width, dims.height),
            params=result.params.to_dict(),
            action=result.action,
            diagnostics=result.diagnostics,
            image_width=dims.width,
            image_height=dims.height,
        )

    # ------------------------------------------------------------------
    # Errors
    # ------------------------------------------------------------------

    @app.exception_handler(TaskSourceError)
    def _task_source_error(_: Request, exc: TaskSourceError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)})


# --------------------------------------------------------------------------
# Serving helpers
# --------------------------------------------------------------------------

def _resolve_annotator(
    payload: LSPredictRequest, header_value: Optional[str], settings: Settings
) -> str:
    """
    Work out who is being served.

    Auth is the MVP header stub the plan specifies, not real identity. Label
    Studio does not forward an annotator header to an ML backend, so in practice
    the id comes from `params.context.user` when Label Studio supplies it, and
    otherwise falls back to a placeholder.

    That matters downstream: Dev 2's honeypot trust scores are keyed on
    `annotator_id`, so a placeholder id makes every annotator look like the same
    person. Recorded as D16 - it needs real auth before trust scoring means
    anything, which the plan already defers.
    """
    if header_value:
        return header_value

    context = payload.params.get("context") or {}
    for key in ("annotator_id", "user", "username", "email"):
        value = context.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for key in ("login", "user"):
        value = payload.params.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return settings.default_annotator_id


def _box_of(task: QueueTask) -> List[float]:
    box = task.bounding_box
    return [box.x_min, box.y_min, box.x_max, box.y_max]


def _claim_task(app: FastAPI, queue: str, annotator_id: str) -> QueueTask:
    return app.state.task_source.next_task(queue=queue, annotator_id=annotator_id)


def _predict_one(
    app: FastAPI,
    ls_task: LSTask,
    info: ls_format.LabelConfigInfo,
    annotator_id: str,
    ls_project_id: Optional[str],
    settings: Settings,
):
    """One task's prediction. Never raises - see the note on /predict."""
    try:
        task = _claim_task(app, settings.junior_queue_name, annotator_id)
    except NoTaskAvailable:
        log.info("No pending task for annotator %s; returning an empty prediction.", annotator_id)
        return ls_format.empty_prediction(settings)
    except TaskSourceError as exc:
        log.error("Could not claim a task for %s: %s", annotator_id, exc)
        return ls_format.empty_prediction(settings)

    dims: ImageDims = app.state.dimensions.resolve(task.image_url, task_data=ls_task.data)

    if not dims.reliable:
        # Serving here would put the polygon at the wrong scale and in the wrong
        # place. The annotator would correct it anyway, and the effort telemetry
        # from that correction would describe a mask the policy never proposed -
        # a plausible-looking number that silently corrupts the reward. Losing
        # one rollout is the cheaper failure. See image_meta.ImageDims.reliable.
        log.error(
            "Refusing to serve task %s: image dimensions for %s could not be determined. "
            "The annotator will draw from scratch and this task produces no rollout.",
            task.task_id, task.image_url,
        )
        return ls_format.empty_prediction(settings)

    try:
        result = apply_wiggle(
            task.baseline_mask.points,
            seed=new_seed(settings.wiggle_seed_bytes),
            sigma=settings.wiggle_sigma,
            mode=settings.wiggle_mode,
            scale_reference=settings.wiggle_scale_reference,
            bounding_box=_box_of(task),
            image_width=dims.width,
            image_height=dims.height,
            clamp=settings.wiggle_clamp_to_frame,
        )
    except WiggleError as exc:
        log.error(
            "Cannot wiggle the baseline mask for task %s (%s). Serving an empty prediction; "
            "the annotator will draw from scratch and this task produces no usable rollout.",
            task.task_id, exc,
        )
        return ls_format.empty_prediction(settings)

    bundle = ls_format.build_prediction(task, result, dims, settings, info)

    app.state.store.record(
        ServedWiggleRecord(
            task_id=task.task_id,
            image_id=task.image_id,
            wiggle_seed=result.params.seed,
            served_at=utc_now_iso(),
            baseline_points=result.baseline_points,
            wiggled_points=result.points,
            wiggle_params=result.params.to_dict(),
            action=result.action,
            diagnostics=result.diagnostics,
            image_width=dims.width,
            image_height=dims.height,
            image_dims_source=dims.source,
            annotator_id=annotator_id,
            ls_project_id=ls_project_id,
            ls_task_id=ls_task.id,
            model_version=settings.model_version,
            queue=task.queue.value,
            is_honeypot=task.honeypot.is_honeypot,
        )
    )

    log.info(
        "Served task %s to %s (seed=%s, iou_vs_baseline=%.3f, mean_shift=%.1fpx)",
        task.task_id, annotator_id, result.params.seed,
        result.diagnostics["iou_vs_baseline"], result.diagnostics["mean_displacement_px"],
    )
    return bundle.prediction


app = create_app()
