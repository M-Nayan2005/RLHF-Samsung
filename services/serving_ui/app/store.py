"""
Persistence for the polygon this service actually served.

**Why this exists.** `docs/reference/open-questions.md` Q11 (and divergence D10)
flags the most consequential gap in the frozen contracts: both schemas label the
*pre-wiggle* mask as `M_initial`, but Tier 3's `delta-IoU` has to be computed
against the mask the human actually corrected - the *wiggled* one. In RL terms
the wiggled polygon is the sampled action `A_t`, and reward must attribute to the
action taken, not to the mean of the distribution it was sampled from. That
polygon is persisted nowhere in any frozen contract.

`wiggle_seed` makes it *reconstructible*, and `wiggle.py` goes to some trouble to
keep the reconstruction exact. But reconstruction stays correct only while the
RNG, the vertex ordering and the transform all remain pinned, and a reward signal
resting on three invariants holding across every future refactor is a fragile
thing to build a training loop on. Writing the polygon down costs one JSONL
append per served task.

**This does not resolve Q11.** Q11 asks what `M_initial` *means*, which is a
decision for whoever owns Tier 3, and the answer may still be "the consensus
mask". This only guarantees that whichever answer is chosen, the data to
implement it exists. Logged as D15.

**Storage.** JSONL by default: append-only, greppable, survives a container
restart through the mounted volume, and needs no table that Dev 1 and Dev 2 own
the migrations for. A Postgres backend belongs here eventually - `served_wiggles`
would be the natural table - but adding one tonight would mean coordinating a
migration across three tracks for data nothing reads yet.
"""
from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterator, Optional

from .config import Settings
from .models import ServedWiggleRecord

log = logging.getLogger(__name__)

# How many recent records to keep indexed in memory for /served lookups. The
# JSONL file is the durable record; this is only a read cache, and an annotator
# session that outruns it falls back to a file scan.
_INDEX_CAPACITY = 4096


class ServedWiggleStore:
    """
    Append-only store of served wiggles, indexed by `task_id` and `wiggle_seed`.

    Thread-safe: uvicorn serves `/predict` from a worker thread pool, and two
    annotators pulling tasks at the same moment would otherwise interleave
    partial lines in the JSONL.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._lock = threading.Lock()
        self._by_task: "OrderedDict[str, ServedWiggleRecord]" = OrderedDict()
        self._by_seed: "OrderedDict[str, ServedWiggleRecord]" = OrderedDict()
        self._path: Optional[Path] = None

        if settings.served_store_backend == "jsonl":
            self._path = Path(settings.served_store_path)
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                # Do not take the service down over this. A served task with no
                # audit row is recoverable from the seed; a service that refuses
                # to serve is not.
                log.error(
                    "Cannot create the served-wiggle directory %s (%s). Falling back to "
                    "in-memory only: served polygons will NOT survive a restart.",
                    self._path.parent, exc,
                )
                self._path = None

    # ---------------------------------------------------------------- write

    def record(self, record: ServedWiggleRecord) -> None:
        with self._lock:
            self._remember(record)
            if self._path is None:
                return
            try:
                with self._path.open("a", encoding="utf-8") as fh:
                    fh.write(record.model_dump_json() + "\n")
            except OSError as exc:
                log.error(
                    "Failed to persist the served wiggle for task %s (%s). The polygon is "
                    "still reconstructible from seed %s.",
                    record.task_id, exc, record.wiggle_seed,
                )

    def _remember(self, record: ServedWiggleRecord) -> None:
        # A task re-served after a consensus requeue gets a fresh seed, and the
        # newest serve is the one whose telemetry is arriving, so overwrite.
        self._by_task[record.task_id] = record
        self._by_task.move_to_end(record.task_id)
        self._by_seed[record.wiggle_seed] = record
        self._by_seed.move_to_end(record.wiggle_seed)

        while len(self._by_task) > _INDEX_CAPACITY:
            self._by_task.popitem(last=False)
        while len(self._by_seed) > _INDEX_CAPACITY:
            self._by_seed.popitem(last=False)

    # ----------------------------------------------------------------- read

    def by_task(self, task_id: str) -> Optional[ServedWiggleRecord]:
        with self._lock:
            hit = self._by_task.get(task_id)
        return hit or self._scan(lambda r: r.task_id == task_id)

    def by_seed(self, wiggle_seed: str) -> Optional[ServedWiggleRecord]:
        with self._lock:
            hit = self._by_seed.get(wiggle_seed)
        return hit or self._scan(lambda r: r.wiggle_seed == wiggle_seed)

    def _scan(self, predicate) -> Optional[ServedWiggleRecord]:
        """
        Last matching record in the file.

        Linear, and that is acceptable: it only runs on a cache miss, which means
        a lookup for a task served long enough ago to have aged out of the index.
        If this ever becomes hot, that is the signal to move the store to
        Postgres rather than to grow the cache.
        """
        found = None
        for record in self.iter_records():
            if predicate(record):
                found = record
        return found

    def iter_records(self) -> Iterator[ServedWiggleRecord]:
        if self._path is None or not self._path.exists():
            return
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                for line_no, line in enumerate(fh, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield ServedWiggleRecord.model_validate_json(line)
                    except ValueError as exc:
                        log.warning(
                            "Skipping malformed served-wiggle record at %s:%s (%s)",
                            self._path, line_no, exc,
                        )
        except OSError as exc:
            log.error("Cannot read the served-wiggle store at %s: %s", self._path, exc)

    # ----------------------------------------------------------------- meta

    def stats(self) -> Dict[str, object]:
        with self._lock:
            indexed = len(self._by_task)
        return {
            "backend": self._settings.served_store_backend,
            "path": str(self._path) if self._path else None,
            "durable": self._path is not None,
            "indexed_tasks": indexed,
        }
