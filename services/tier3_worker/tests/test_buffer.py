"""
Buffer-level details that only bite against a real driver.

`as_timestamptz` exists because of a failure the in-memory double cannot
reproduce: `ExperienceTuple.created_at` is a string, every timestamp in the
frozen contracts is a string, and asyncpg refuses a string for a `timestamptz`
parameter. It encodes arguments client-side, so the `$13::timestamptz` cast in
the INSERT never gets a chance to help. Caught by running
`python -m tier3_worker.selfcheck` against a live Postgres; pinned here so it
stays caught offline.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tier3_worker.aggregator import utc_now_iso
from tier3_worker.buffer import as_timestamptz


def test_the_aggregators_own_output_round_trips():
    """The realistic case: whatever `build_experience_tuple` stamps must bind."""
    parsed = as_timestamptz(utc_now_iso())
    assert isinstance(parsed, datetime)
    assert parsed.tzinfo is not None
    assert abs((datetime.now(timezone.utc) - parsed).total_seconds()) < 5


def test_trailing_z_is_understood():
    parsed = as_timestamptz("2026-08-24T16:43:09.712Z")
    assert parsed == datetime(2026, 8, 24, 16, 43, 9, 712000, tzinfo=timezone.utc)


def test_explicit_offset_is_preserved():
    parsed = as_timestamptz("2026-08-24T18:43:09+02:00")
    assert parsed.utcoffset() == timedelta(hours=2)


def test_a_naive_timestamp_is_read_as_utc():
    """
    Not as local time. Every producer here emits UTC, and attaching the worker's
    zone to a timestamp that is really UTC would skew `created_at` ordering —
    which is exactly what Tier 4 batches on.
    """
    assert as_timestamptz("2026-08-24T16:43:09").tzinfo == timezone.utc


def test_a_datetime_passes_through():
    now = datetime.now(timezone.utc)
    assert as_timestamptz(now) is now


def test_a_naive_datetime_gains_utc():
    assert as_timestamptz(datetime(2026, 8, 24, 16, 43, 9)).tzinfo == timezone.utc


def test_garbage_raises_a_named_error():
    with pytest.raises(ValueError, match="not an ISO-8601 timestamp"):
        as_timestamptz("last tuesday")
