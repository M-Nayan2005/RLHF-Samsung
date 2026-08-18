"""
Where a `QueueTask` comes from.

Two implementations behind one interface:

* `RoutingQATaskSource` - the real thing. Calls Dev 2's
  `GET /tasks/next?queue=junior&annotator_id=...`, which atomically claims the
  oldest pending task and flips it to `assigned`.
* `MockTaskSource` - reads `tests/mocks/routing_task.json`. Active when
  `MOCK_MODE=true`, so this track can be built and demoed before `routing_qa`
  exists. Named in the Dev 3 brief as the local-testing path.

One thing this module deliberately does *not* do: fall back to the mock when
routing_qa is unreachable. A silent fallback would mean an integration test
"passing" against a fixture while the real upstream is down, and every rollout
in that batch would carry the fixture's geometry. `MOCK_MODE` is the only way to
get fixture data, and it is loud about it at boot.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Protocol

import httpx
from pydantic import ValidationError

from common.schemas.routing_queue import QueueTask

from .config import Settings

log = logging.getLogger(__name__)


class TaskSourceError(RuntimeError):
    """Upstream could not supply a task. Carries whether it is worth retrying."""

    def __init__(self, message: str, *, status_code: int = 502, retryable: bool = True):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class NoTaskAvailable(TaskSourceError):
    """The queue is empty. Not an error condition - Label Studio just gets no prediction."""

    def __init__(self, queue: str):
        super().__init__(f"no pending task in {queue}", status_code=404, retryable=False)


class TaskSource(Protocol):
    def next_task(self, *, queue: str, annotator_id: str) -> QueueTask: ...
    def describe(self) -> str: ...


# --------------------------------------------------------------------------
# Real upstream
# --------------------------------------------------------------------------

class RoutingQATaskSource:
    def __init__(self, settings: Settings, client: Optional[httpx.Client] = None):
        self._settings = settings
        self._client = client or httpx.Client(timeout=settings.request_timeout_s)

    def describe(self) -> str:
        return f"routing_qa @ {self._settings.routing_qa_url}"

    def next_task(self, *, queue: str, annotator_id: str) -> QueueTask:
        url = f"{self._settings.routing_qa_url}/tasks/next"
        params = {"queue": queue, "annotator_id": annotator_id}
        headers = {self._settings.annotator_id_header: annotator_id}

        try:
            response = self._client.get(url, params=params, headers=headers)
        except httpx.RequestError as exc:
            raise TaskSourceError(f"routing_qa unreachable at {url}: {exc}") from exc

        if response.status_code == 404:
            raise NoTaskAvailable(queue)
        if response.status_code >= 500:
            raise TaskSourceError(
                f"routing_qa returned {response.status_code} for {url}", retryable=True
            )
        if response.status_code >= 400:
            raise TaskSourceError(
                f"routing_qa rejected the claim with {response.status_code}: {response.text[:400]}",
                status_code=502,
                retryable=False,
            )

        return _parse_task(response.json(), source=url)


# --------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------

class MockTaskSource:
    """
    Serves one canned `QueueTask` forever.

    The fixture is re-read on every call rather than cached, so the file can be
    edited between requests to test a different geometry without a restart.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._path = Path(settings.mock_task_path)

    def describe(self) -> str:
        return f"MOCK_MODE fixture @ {self._path}"

    def next_task(self, *, queue: str, annotator_id: str) -> QueueTask:
        if not self._path.exists():
            raise TaskSourceError(
                f"MOCK_MODE=true but no fixture at {self._path}. "
                f"Set MOCK_TASK_PATH, or copy tests/mocks/routing_task.json there.",
                status_code=500,
                retryable=False,
            )
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise TaskSourceError(
                f"fixture at {self._path} is not valid JSON: {exc}",
                status_code=500,
                retryable=False,
            ) from exc

        task = _parse_task(raw, source=str(self._path))

        # The fixture ships with status "pending" and assigned_to null, because
        # it is a snapshot of what Dev 2 writes to the queue table. The real
        # endpoint claims the task before returning it, so mirror that here -
        # otherwise mock and live modes hand /predict differently-shaped state.
        return task.model_copy(update={"status": "assigned", "assigned_to": annotator_id})


def _parse_task(raw: dict, *, source: str) -> QueueTask:
    """
    Validate against the frozen `QueueTask` contract.

    A `ValidationError` here means Dev 2's output drifted from
    `common/schemas/routing_queue.py`. That is worth a loud, specific error
    rather than a generic 500, because the same failure will hit Dev 4 one hop
    later and the two of us should not debug it twice.
    """
    try:
        task = QueueTask.model_validate(raw)
    except ValidationError as exc:
        log.error(
            "QueueTask from %s failed contract validation: %s",
            source,
            exc.errors(include_url=False),
        )
        raise TaskSourceError(
            f"task from {source} does not match the frozen QueueTask contract: "
            f"{exc.error_count()} field error(s); see logs",
            status_code=502,
            retryable=False,
        ) from exc

    if task.honeypot.ground_truth_mask is not None:
        # Dev 2 is required to strip this before serialising. If it arrives here
        # it is a leak of the answer key into a process that talks to a browser.
        log.error(
            "SECURITY: task %s arrived from %s with honeypot.ground_truth_mask populated. "
            "routing_qa must strip it. Dropping the field before it can reach the frontend.",
            task.task_id,
            source,
        )
        task = task.model_copy(
            update={"honeypot": task.honeypot.model_copy(update={"ground_truth_mask": None})}
        )

    return task


def build_task_source(settings: Settings, client: Optional[httpx.Client] = None) -> TaskSource:
    if settings.mock_mode:
        return MockTaskSource(settings)
    return RoutingQATaskSource(settings, client=client)
