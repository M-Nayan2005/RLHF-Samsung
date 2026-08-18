import os
import sys
import json
import fakeredis
from fastapi.testclient import TestClient

# Make sure we can import services and common
sys.path.insert(0, os.path.abspath("."))

from services.webhook_gateway import main

# Mock Redis in the app
fake_redis = fakeredis.FakeRedis(decode_responses=True)
main.redis_client = fake_redis
main.REDIS_TELEMETRY_QUEUE_KEY = "telemetry:ingest"

# Initialize TestClient
client = TestClient(main.app)

def run_tests():
    print("=== Testing Webhook Gateway End-to-End ===")

    # 1. Test standard webhook endpoint
    print("\n[1] Testing POST /webhooks/label-studio")
    with open("tests/mocks/ls_webhook_payload.json", "r") as f:
        ls_payload = json.load(f)
    
    response = client.post(
        "/webhooks/label-studio",
        json=ls_payload,
        headers={"Authorization": "default_secret"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # 2. Test Idempotency (sending same payload again)
    print("\n[2] Testing Idempotency (Duplicate POST)")
    response = client.post(
        "/webhooks/label-studio",
        json=ls_payload,
        headers={"Authorization": "default_secret"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # 3. Test raw telemetry fallback endpoint
    print("\n[3] Testing POST /telemetry/raw")
    with open("tests/mocks/telemetry_raw_payload.json", "r") as f:
        raw_payload = json.load(f)
    
    response = client.post(
        "/telemetry/raw",
        json=raw_payload
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # 4. Consumer verification (checking Redis queue)
    print("\n[4] Verifying Redis Queue (Simulating Consumer)")
    queue_len = fake_redis.llen(main.REDIS_TELEMETRY_QUEUE_KEY)
    print(f"Events in queue: {queue_len}")
    
    while True:
        event = fake_redis.rpop(main.REDIS_TELEMETRY_QUEUE_KEY)
        if not event:
            break
        envelope = json.loads(event)
        print(f"-> Consumed Event: {envelope['event_id']}")
        print(f"   Idempotency Key: {envelope['idempotency_key']}")
        print(f"   Payload Task ID: {envelope['payload']['task_id']}")

    print("\n=== All Tests Finished ===")

if __name__ == "__main__":
    run_tests()
