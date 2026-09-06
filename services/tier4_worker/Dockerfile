FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if any
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY services/tier4_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy common schemas
COPY common/ /app/common/

# Copy tier4 worker code
COPY services/tier4_worker/ /app/tier4_worker/

ENV PYTHONPATH=/app

CMD ["python", "-m", "tier4_worker.trainer"]
