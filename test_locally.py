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

    # 1. Test Beacon Endpoint
    print("\n[1] Testing POST /telemetry/raw (Beacon)")
    with open("tests/mocks/telemetry_raw_payload.dev3.json", "r") as f:
        raw_payload_text = f.read()
    
    response = client.post(
        "/telemetry/raw",
        content=raw_payload_text,
        headers={"Content-Type": "text/plain;charset=UTF-8"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # 2. Test Webhook Endpoint (Merge)
    print("\n[2] Testing POST /webhooks/label-studio")
    with open("tests/mocks/ls_webhook_payload.json", "r") as f:
        ls_payload = json.load(f)
    
    # Simulate the real webhook by stripping the telemetry and adding the seed string
    if "effort_telemetry" in ls_payload:
        del ls_payload["effort_telemetry"]
    
    # Inject wiggle_seed into the raw JSON to simulate Dev 3's string encoding
    ls_payload_text = json.dumps(ls_payload)
    # The wiggle seed from dev3 payload is "c644b8cf379b545910d076f8e05d913c"
    ls_payload_text = ls_payload_text.replace('"task_a1b2c3"', '"task_a1b2c3", "model_version": "serving-ui-stochastic-0.1.0|seed=c644b8cf379b545910d076f8e05d913c"')

    response = client.post(
        "/webhooks/label-studio",
        content=ls_payload_text,
        headers={"X-Label-Studio-Secret": "default_secret", "Content-Type": "application/json"}
    )
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")

    # 3. Test Webhook Endpoint without Beacon (Fallback)
    print("\n[3] Testing POST /webhooks/label-studio (No Beacon Fallback)")
    # Change the seed to something not in Redis
    ls_payload_text_fallback = ls_payload_text.replace("c644b8cf379b545910d076f8e05d913c", "missing_seed_123")
    
    # Change annotation_id to avoid idempotency block
    ls_payload_text_fallback = ls_payload_text_fallback.replace("ann_9f8e7d", "ann_fallback123")

    response = client.post(
        "/webhooks/label-studio",
        content=ls_payload_text_fallback,
        headers={"X-Label-Studio-Secret": "default_secret", "Content-Type": "application/json"}
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
        print(f"   Merged Telemetry Click Count: {envelope['payload']['effort_telemetry']['click_count']}")

    print("\n=== All Tests Finished ===")

if __name__ == "__main__":
    run_tests()
