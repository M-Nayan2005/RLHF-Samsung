import uuid
import random
from datetime import datetime, timezone
from config import settings

def route_prediction(prediction: dict) -> dict:
    variance = prediction['geometric_variance']
    entropy = prediction['class_logit_entropy']
    
    queue = "junior_queue"
    reason = "low_entropy_low_variance"
    
    # Extreme combined score -> consensus
    # Thresholds are examples; we use config settings here.
    # E.g. variance > 2x threshold AND entropy > 2x threshold
    if variance >= (settings.VARIANCE_THRESHOLD * 2) and entropy >= (settings.ENTROPY_THRESHOLD * 2):
        queue = "consensus_queue"
        reason = "extreme_combined_score"
    elif entropy >= settings.ENTROPY_THRESHOLD:
        queue = "senior_queue"
        reason = "high_entropy_confidently_wrong"
    elif variance >= settings.VARIANCE_THRESHOLD:
        queue = "senior_queue"
        reason = "high_variance_messy_mask"
    
    # Stochastic Audit Filter
    if queue == "junior_queue" and random.random() < settings.STOCHASTIC_AUDIT_RATE:
        queue = "consensus_queue"
        reason = "stochastic_audit_filter"
        
    return {
        "queue": queue,
        "routing_metrics": {
            "geometric_variance": variance,
            "class_logit_entropy": entropy,
            "routed_reason": reason
        }
    }

def maybe_inject_honeypot() -> dict:
    if random.random() < settings.HONEYPOT_INJECTION_RATE:
        return {
            "is_honeypot": True,
            "ground_truth_mask": {
                "points": [[0,0], [10,0], [10,10], [0,10]] # Placeholder known-gold mask
            }
        }
    return {"is_honeypot": False, "ground_truth_mask": None}

def build_queue_task(prediction: dict) -> dict:
    routing_info = route_prediction(prediction)
    honeypot_info = maybe_inject_honeypot() if routing_info["queue"] == "junior_queue" else {"is_honeypot": False, "ground_truth_mask": None}
    
    return {
        "task_id": f"task_{uuid.uuid4().hex[:8]}",
        "image_id": prediction["image_id"],
        "image_url": prediction["image_url"],
        "bounding_box": prediction["bounding_box"],
        "baseline_mask": prediction["consensus_mask"],
        "queue": routing_info["queue"],
        "routing_metrics": routing_info["routing_metrics"],
        "honeypot": honeypot_info,
        "retry_count": 0,
        "status": "pending",
        "assigned_to": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
