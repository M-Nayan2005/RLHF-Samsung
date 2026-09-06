# tier3_worker — Tier 3 reward pipeline.
#
# Build context is the REPO ROOT, not this directory, because the image needs
# `common/schemas/` alongside the service code. docker-compose.yml sets:
#
#     build:
#       context: .
#       dockerfile: services/tier3_worker/Dockerfile
#
# Building from inside services/tier3_worker/ will fail on the `COPY common`
# line, which is the intended failure — a worker that cannot import the frozen
# contracts should not start.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# shapely ships manylinux wheels with GEOS bundled, so no apt build stage is
# needed on this base image. If that ever stops being true the failure is a
# loud build error, not a runtime surprise.
COPY services/tier3_worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The frozen contracts. `tier3_rlhf.py` is Tier 3-owned and additive; the four
# Tier 1/2 modules beside it are read-only by convention.
COPY common /app/common

# Service code lands at /app/tier3_worker so `from tier3_worker.stage2 import
# process_stage1` and `from common.schemas... import ...` both resolve with
# /app on sys.path.
COPY services/tier3_worker /app/tier3_worker

# The DDL, so the container can apply or verify its own schema.
COPY infra/sql /app/infra/sql

# Verifies config, connectivity and schema, then exits. Safe to run against a
# live database — every statement in the DDL is idempotent.
#   docker compose run --rm tier3_worker python -m tier3_worker.selfcheck
#
# The default command is Dev 1's consumer entrypoint, which owns the Redis
# BRPOP loop and calls `stage2.process_stage1`. Until it lands, the compose
# service sits behind the `tier3` profile and does not start.
CMD ["python", "-m", "tier3_worker.consumer"]
