"""
Tier 3 worker settings.

Dev 2 owns the reward-side keys (`TIER3_ALPHA`, `TIER3_BETA`,
`TIER3_REFERENCE_MASK`) and the Postgres DSN. Dev 1's consumer keys — Redis
URL, queue name, worker fan-out, the bot-velocity ceiling — belong in this same
file when they land; the sections are kept separate so the two additions merge
without touching each other's lines.

**On alpha and beta.** `AGENTS.md` forbids inventing an unset hyperparameter,
and `docs/reference/equations.md` is explicit that alpha, beta and w1..w3 "are
never assigned anywhere" in the source documents. The values here are therefore
*not* derived from the spec — they are the starting points the Tier 3/4
implementation plan nominates ("tune later, not tonight"). They are read from
the environment rather than hardcoded into the arithmetic, and
`log_provisional_hyperparameters()` announces them at startup so no run is ever
silently calibrated. Recorded as open question Q18 in the workspace ledger.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Set

log = logging.getLogger(__name__)

# Which mask plays the role of `M_initial` in the delta-IoU. See open question
# Q11 / divergence D10 — the frozen schemas label the consensus mask
# `M_initial`, but the reward has to attribute to the action the policy
# actually took, which is the wiggled polygon the human corrected.
REFERENCE_WIGGLED = "wiggled"
REFERENCE_INITIAL = "initial"
REFERENCE_CHOICES: Set[str] = {REFERENCE_WIGGLED, REFERENCE_INITIAL}


def _get(name: str, default: str) -> str:
    value = os.environ.get(name)
    return default if value is None or value.strip() == "" else value.strip()


def _get_float(name: str, default: float) -> float:
    raw = _get(name, str(default))
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; falling back to %s", name, raw, default)
        return default


def _get_int(name: str, default: int) -> int:
    raw = _get(name, str(default))
    try:
        return int(raw)
    except ValueError:
        log.warning("%s=%r is not an integer; falling back to %s", name, raw, default)
        return default


def _get_bool(name: str, default: bool) -> bool:
    return _get(name, "true" if default else "false").lower() in {"1", "true", "yes", "on"}


def normalize_dsn(dsn: str) -> str:
    """
    Strip a SQLAlchemy dialect suffix off a DSN.

    `.env.example` ships `DATABASE_URL=postgresql+asyncpg://...`, which is
    SQLAlchemy's spelling. `asyncpg.create_pool` parses the scheme itself and
    rejects anything with a `+driver` on it, so a worker inheriting the shared
    variable would fail to connect for a reason that has nothing to do with its
    credentials. Normalising here is cheaper than asking every deployment to
    keep a second, near-identical URL in sync.
    """
    scheme, separator, rest = dsn.partition("://")
    if not separator or "+" not in scheme:
        return dsn
    return f"{scheme.split('+', 1)[0]}://{rest}"


def _get_choice(name: str, default: str, choices: Set[str]) -> str:
    value = _get(name, default).lower()
    if value not in choices:
        log.warning(
            "%s=%r is not one of %s; falling back to %r",
            name, value, sorted(choices), default,
        )
        return default
    return value


@dataclass(frozen=True)
class Settings:
    # -- E-DRDE reward weights (provisional, see module docstring) ---------
    alpha: float
    beta: float

    # -- geometry ---------------------------------------------------------
    reference_mask: str
    use_gold_when_available: bool

    # -- persistence ------------------------------------------------------
    pg_dsn: str
    db_statement_timeout_ms: int
    apply_schema_on_start: bool

    # -- behaviour --------------------------------------------------------
    require_model_version: bool

    @property
    def reference_is_wiggled(self) -> bool:
        return self.reference_mask == REFERENCE_WIGGLED


def load_settings() -> Settings:
    return Settings(
        alpha=_get_float("TIER3_ALPHA", 1.0),
        beta=_get_float("TIER3_BETA", 0.3),
        reference_mask=_get_choice("TIER3_REFERENCE_MASK", REFERENCE_WIGGLED, REFERENCE_CHOICES),
        use_gold_when_available=_get_bool("TIER3_USE_GOLD_WHEN_AVAILABLE", True),
        pg_dsn=normalize_dsn(
            _get(
                "TIER3_PG_DSN",
                # Falls back to the shared variable, then to the compose defaults
                # in docker-compose.yml (user/password/rlhf_seg).
                os.environ.get("DATABASE_URL")
                or "postgresql://user:password@postgres:5432/rlhf_seg",
            )
        ),
        db_statement_timeout_ms=_get_int("TIER3_DB_STATEMENT_TIMEOUT_MS", 5000),
        apply_schema_on_start=_get_bool("TIER3_APPLY_SCHEMA_ON_START", True),
        require_model_version=_get_bool("TIER3_REQUIRE_MODEL_VERSION", True),
    )


def log_provisional_hyperparameters(settings: Settings) -> None:
    """
    Announce that the reward weights are uncalibrated.

    Deliberately WARNING, not INFO. A reward scale nobody chose on purpose is
    the single easiest way to end up with a trained model that is confidently
    optimising the wrong thing, and the sources genuinely never assign these.
    """
    log.warning(
        "E-DRDE weights are PROVISIONAL and not spec-derived: alpha=%s beta=%s "
        "(reference_mask=%s, gold_path=%s). The source documents never assign "
        "alpha/beta/w1..w3 — see docs/reference/equations.md. Calibrate before "
        "any run whose output is trusted.",
        settings.alpha, settings.beta, settings.reference_mask,
        "on" if settings.use_gold_when_available else "off",
    )
