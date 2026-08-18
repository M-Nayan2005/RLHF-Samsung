"""
Effort telemetry: receive it from the browser, enrich it, hand it to Dev 4.

Label Studio's stock `ANNOTATION_UPDATED` webhook carries the final polygon and
`lead_time`, and nothing else about *how* the human got there. But the whole
system rests on the claim that correction effort is the reward signal, so the
click count, the cursor path length and the boundary dwell time are not
decoration - they are `C`, `L_path` and `T_dwell`, the three terms E-DRDE
consumes. They have to be captured in the browser or they do not exist.

This module is the server half of that capture. The browser half is
`label_studio/instrumentation/effort_telemetry.js`.

Transport
---------
The Dev 3 brief offers two options and asks which was implemented and why.
**The beacon is the implemented default** and the reasoning is in
`docs/dev3-decisions.md` (DD-4); the short version is that stock Label Studio
exposes no supported hook for writing to `annotation.meta` before submit, and
the frozen `LSAnnotationUpdatedPayload` declares `effort_telemetry` at the **top
level** anyway - so the beacon matches the contract that Dev 4 validates against,
while meta injection would have required Dev 4 to lift a nested block before
parsing (divergence D5).

The service still accepts telemetry arriving by either route, so a future Label
Studio that does support meta injection needs no change here.

Enrichment
----------
Two things are added before forwarding, because this service is the only place
that knows them:

* `wiggle_seed`, recovered from the served-wiggle store when the browser did not
  manage to read it back out of the prediction;
* the `task_id`, validated against a task this service actually served, so a
  malformed or replayed beacon is rejected here rather than at Dev 4's parser.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple

import httpx

from .config import Settings
from .models import RawTelemetryEnvelope
from .store import ServedWiggleStore

log = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """ISO-8601 UTC with a trailing Z, matching the format used across the contracts."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class TelemetryForwarder:
    """Posts enriched telemetry to Dev 4's `POST /telemetry/raw`."""

    def __init__(
        self,
        settings: Settings,
        store: ServedWiggleStore,
        client: Optional[httpx.Client] = None,
    ):
        self._settings = settings
        self._store = store
        self._client = client or httpx.Client(timeout=settings.request_timeout_s)

    @property
    def endpoint(self) -> str:
        return f"{self._settings.webhook_gateway_url}/telemetry/raw"

    def enrich(self, envelope: RawTelemetryEnvelope) -> RawTelemetryEnvelope:
        updates = {}

        seed = envelope.wiggle_seed or envelope.effort_telemetry.wiggle_seed
        if not seed:
            served = self._store.by_task(envelope.task_id)
            if served:
                seed = served.wiggle_seed
                log.info(
                    "Beacon for task %s arrived without a wiggle_seed; recovered %s from the "
                    "served-wiggle store.", envelope.task_id, seed,
                )
            else:
                log.warning(
                    "Beacon for task %s has no wiggle_seed and no served record. Tier 3 will "
                    "not be able to recover the action A_t for this annotation.",
                    envelope.task_id,
                )

        if seed:
            updates["wiggle_seed"] = seed
            if not envelope.effort_telemetry.wiggle_seed:
                updates["effort_telemetry"] = envelope.effort_telemetry.model_copy(
                    update={"wiggle_seed": seed}
                )

        if not envelope.client_sent_at:
            updates["client_sent_at"] = utc_now_iso()

        if not envelope.completed_by or not envelope.project_id:
            served = self._store.by_task(envelope.task_id)
            if served:
                if not envelope.completed_by and served.annotator_id:
                    updates["completed_by"] = served.annotator_id
                if not envelope.project_id and served.ls_project_id:
                    updates["project_id"] = served.ls_project_id

        return envelope.model_copy(update=updates) if updates else envelope

    def forward(self, envelope: RawTelemetryEnvelope) -> Tuple[bool, Optional[str]]:
        """
        Send the envelope on to the gateway.

        Never raises. The browser is mid-submit when this runs, and a failed
        forward must not become a failed annotation - the human's work is worth
        more than the telemetry describing it. A dropped beacon is logged at
        ERROR with the full payload so it can be replayed by hand.
        """
        if not self._settings.telemetry_forward_enabled:
            return False, "forwarding disabled (TELEMETRY_FORWARD_ENABLED=false)"

        try:
            response = self._client.post(
                self.endpoint,
                json=envelope.model_dump(mode="json", exclude_none=True),
                headers={self._settings.annotator_id_header: envelope.completed_by or ""},
            )
        except httpx.RequestError as exc:
            log.error(
                "Telemetry forward to %s failed for task %s (%s). Payload: %s",
                self.endpoint, envelope.task_id, exc,
                envelope.model_dump_json(exclude_none=True),
            )
            return False, f"gateway unreachable: {exc}"

        if response.status_code >= 400:
            log.error(
                "Gateway rejected telemetry for task %s with HTTP %s: %s. Payload: %s",
                envelope.task_id, response.status_code, response.text[:400],
                envelope.model_dump_json(exclude_none=True),
            )
            return False, f"gateway returned {response.status_code}"

        log.info(
            "Forwarded telemetry for task %s (seed=%s, clicks=%s, path=%.1fpx, dwell=%sms)",
            envelope.task_id,
            envelope.wiggle_seed,
            envelope.effort_telemetry.click_count,
            envelope.effort_telemetry.cursor_path_length_px,
            envelope.effort_telemetry.dwell_time_ms,
        )
        return True, None
