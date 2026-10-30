# tier4_worker - PPO trainer for the wiggle policy.
#
# Build context is the REPO ROOT (see docker-compose.yml): the image needs
# common/ (rl_policy + schemas) and infra/sql alongside the service code.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only torch: the policy is tiny, and the default wheel drags in CUDA.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY services/tier4_worker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY common /app/common
COPY infra/sql /app/infra/sql
COPY services/tier4_worker /app/tier4_worker

ENV PYTHONPATH=/app

CMD ["python", "-m", "tier4_worker.trainer"]
