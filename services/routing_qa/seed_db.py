import json
import os
import uuid
from datetime import datetime, timezone
import sys

# Ensure config and db modules can be imported when run directly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from db import init_db, get_db_connection

def seed_tier1_predictions():
    # 1. Initialize DB (creates tier1_predictions placeholder and all queue tables)
    init_db()
    print("Database initialized successfully.")
    
    # 2. Load the mock fixture
    mock_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests", "mocks", "tier1_output.json")
    try:
        with open(mock_path, "r") as f:
            base_mock = json.load(f)
    except Exception as e:
        print(f"Could not load mock fixture from {mock_path}: {e}")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    # 3. Create variants to hit different routing branches
    variants = [
        # Junior route: low variance, low entropy
        {"v": 0.02, "e": 0.1, "desc": "Junior Test"},
        # Senior route (confidently wrong): low variance, high entropy
        {"v": 0.02, "e": 0.4, "desc": "Senior Test (High Entropy)"},
        # Senior route (messy mask): high variance, low entropy
        {"v": 0.15, "e": 0.1, "desc": "Senior Test (High Variance)"},
        # Consensus route: extreme combined (e.g. variance > 0.2 and entropy > 0.5)
        {"v": 0.25, "e": 0.6, "desc": "Consensus Test"},
    ]

    for variant in variants:
        mock = dict(base_mock)
        mock["image_id"] = f"img_{uuid.uuid4().hex[:8]}"
        mock["geometric_variance"] = variant["v"]
        mock["class_logit_entropy"] = variant["e"]
        mock["created_at"] = datetime.now(timezone.utc).isoformat()
        
        cursor.execute("""
            INSERT INTO tier1_predictions (
                image_id, image_url, text_prompt, bounding_box, mcd_samples, 
                geometric_variance, class_logit_entropy, consensus_mask, model_version, created_at
            ) VALUES (
                %(image_id)s, %(image_url)s, %(text_prompt)s, %(bounding_box)s, %(mcd_samples)s,
                %(geometric_variance)s, %(class_logit_entropy)s, %(consensus_mask)s, %(model_version)s, %(created_at)s
            ) ON CONFLICT (image_id) DO NOTHING;
        """, {
            'image_id': mock['image_id'],
            'image_url': mock['image_url'],
            'text_prompt': mock['text_prompt'],
            'bounding_box': json.dumps(mock['bounding_box']),
            'mcd_samples': json.dumps(mock['mcd_samples']),
            'geometric_variance': mock['geometric_variance'],
            'class_logit_entropy': mock['class_logit_entropy'],
            'consensus_mask': json.dumps(mock['consensus_mask']),
            'model_version': mock['model_version'],
            'created_at': mock['created_at']
        })
        print(f"Inserted variant: {variant['desc']} ({mock['image_id']})")

    conn.commit()
    cursor.close()
    conn.close()
    print("Seed complete. Run `SELECT * FROM tier1_predictions;` to verify.")

if __name__ == "__main__":
    seed_tier1_predictions()
