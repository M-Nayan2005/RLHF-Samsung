FROM python:3.11-slim

WORKDIR /app

COPY ./services/webhook_gateway/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy common schemas and the service
COPY ./common /app/common
COPY ./services/webhook_gateway /app/services/webhook_gateway

ENV PYTHONPATH=/app

CMD ["uvicorn", "services.webhook_gateway.main:app", "--host", "0.0.0.0", "--port", "8004"]
