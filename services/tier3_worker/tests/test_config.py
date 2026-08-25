"""Settings loading, and the DSN normalisation that keeps asyncpg connectable."""
from __future__ import annotations

import pytest

from tier3_worker.config import REFERENCE_WIGGLED, load_settings, normalize_dsn


@pytest.mark.parametrize(
    "given, expected",
    [
        # The form .env.example actually ships. asyncpg rejects the `+asyncpg`.
        ("postgresql+asyncpg://u:p@host:5432/db", "postgresql://u:p@host:5432/db"),
        ("postgresql+psycopg2://u:p@host/db", "postgresql://u:p@host/db"),
        # Already clean, and left exactly alone.
        ("postgresql://u:p@host:5432/db", "postgresql://u:p@host:5432/db"),
        ("postgres://u:p@host/db", "postgres://u:p@host/db"),
        # A `+` in the password must not be mistaken for a dialect suffix.
        ("postgresql://u:pa+ss@host/db", "postgresql://u:pa+ss@host/db"),
        ("not-a-dsn", "not-a-dsn"),
    ],
)
def test_normalize_dsn(given, expected):
    assert normalize_dsn(given) == expected


def test_defaults_are_the_plan_values(monkeypatch):
    for key in ("TIER3_ALPHA", "TIER3_BETA", "TIER3_REFERENCE_MASK", "TIER3_PG_DSN", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)

    settings = load_settings()
    assert (settings.alpha, settings.beta) == (1.0, 0.3)
    assert settings.reference_mask == REFERENCE_WIGGLED
    assert settings.require_model_version is True


def test_environment_overrides_are_read(monkeypatch):
    monkeypatch.setenv("TIER3_ALPHA", "2.5")
    monkeypatch.setenv("TIER3_BETA", "0.05")
    monkeypatch.setenv("TIER3_REFERENCE_MASK", "initial")

    settings = load_settings()
    assert (settings.alpha, settings.beta) == (2.5, 0.05)
    assert settings.reference_is_wiggled is False


def test_a_nonsense_reference_mask_falls_back_rather_than_crashing(monkeypatch):
    """A typo in an env var must not silently pick a third behaviour."""
    monkeypatch.setenv("TIER3_REFERENCE_MASK", "wigled")
    assert load_settings().reference_mask == REFERENCE_WIGGLED


def test_a_nonnumeric_weight_falls_back(monkeypatch):
    monkeypatch.setenv("TIER3_ALPHA", "one")
    assert load_settings().alpha == 1.0


def test_shared_database_url_is_inherited_and_normalized(monkeypatch):
    monkeypatch.delenv("TIER3_PG_DSN", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@db:5432/rlhf_seg")
    assert load_settings().pg_dsn == "postgresql://u:p@db:5432/rlhf_seg"
