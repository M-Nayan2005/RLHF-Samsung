# serving_ui — Label Studio ML Backend for Tier 2.
#
# Build context is the REPO ROOT, not this directory, because the image needs
# `common/schemas/` alongside the service code. docker-compose.yml sets:
#
#     build:
#       context: .
#       dockerfile: services/serving_ui/Dockerfile
#
# Building from inside services/serving_ui/ will fail on the `COPY common` line,
# which is the intended failure — a serving_ui that cannot import the frozen
# contracts should not start.

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so a code edit does not re-resolve the whole tree.
COPY services/serving_ui/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# The frozen contracts. Read-only by convention: a schema change is its own PR,
# reviewed by the downstream consumer, per IMPLEMENTATION_PLAN.md §4.
COPY common /app/common

# Service code. `app` lands at /app/app so `from app.main import app` resolves
# and `from common.schemas... import ...` resolves, both with /app on sys.path.
COPY services/serving_ui/app /app/app
COPY services/serving_ui/label_studio /app/label_studio

# Fixtures, for MOCK_MODE. Small, and their absence turns MOCK_MODE into a
# confusing 500 rather than a working demo.
COPY tests/mocks /app/tests/mocks

# Local test images, served at /assets. Copied specifically rather than as
# `COPY tests /app/tests`, because tests/ also holds the hand-dropped Label
# Studio API token during local setup and that must never enter an image.
COPY tests/assets /app/tests/assets

# Durable home for the served-wiggle store (see app/store.py). Mounted as a
# volume in compose; created here so the service still works without one.
RUN mkdir -p /app/data

EXPOSE 8003

# Label Studio polls /health to decide whether the backend is connected.
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8003/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003"]
