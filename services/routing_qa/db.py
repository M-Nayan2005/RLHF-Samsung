import psycopg2
import psycopg2.extras
import json
from config import settings

def get_db_connection():
    return psycopg2.connect(settings.DATABASE_URL)

def init_db():
    conn = get_db_connection()
    conn.autocommit = True
    cursor = conn.cursor()
    
    # Placeholder: Tier 1 Predictions Table (Dev 1's domain, created here for local testing)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tier1_predictions (
            image_id VARCHAR PRIMARY KEY,
            image_url text NOT NULL,
            text_prompt text NOT NULL,
            bounding_box jsonb NOT NULL,
            mcd_samples jsonb NOT NULL,
            geometric_variance float NOT NULL,
            class_logit_entropy float NOT NULL,
            consensus_mask jsonb NOT NULL,
            model_version text NOT NULL,
            created_at timestamp NOT NULL,
            routed boolean DEFAULT FALSE
        );
    """)

    # Dev 2: Queues and state tables
    for queue_table in ["junior_queue", "senior_queue", "consensus_queue", "discard_bin"]:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {queue_table} (
                task_id VARCHAR PRIMARY KEY,
                image_id VARCHAR NOT NULL,
                image_url text NOT NULL,
                bounding_box jsonb NOT NULL,
                baseline_mask jsonb NOT NULL,
                queue VARCHAR NOT NULL,
                routing_metrics jsonb NOT NULL,
                honeypot jsonb NOT NULL,
                retry_count int DEFAULT 0,
                status VARCHAR DEFAULT 'pending',
                assigned_to VARCHAR,
                created_at timestamp,
                updated_at timestamp
            );
        """)
        
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS annotator_trust_scores (
            annotator_id VARCHAR PRIMARY KEY,
            trust_score float DEFAULT 1.0,
            honeypots_passed int DEFAULT 0,
            honeypots_failed int DEFAULT 0
        );
    """)
    
    cursor.close()
    conn.close()

def get_unrouted_predictions():
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cursor.execute("SELECT * FROM tier1_predictions WHERE routed = FALSE FOR UPDATE SKIP LOCKED LIMIT 50;")
    rows = cursor.fetchall()
    
    if rows:
        ids = [row['image_id'] for row in rows]
        cursor.execute("UPDATE tier1_predictions SET routed = TRUE WHERE image_id = ANY(%s)", (ids,))
        conn.commit()
        
    cursor.close()
    conn.close()
    return rows

def insert_queue_task(task: dict):
    conn = get_db_connection()
    cursor = conn.cursor()
    table_name = task['queue']
    
    cursor.execute(f"""
        INSERT INTO {table_name} (
            task_id, image_id, image_url, bounding_box, baseline_mask, queue, 
            routing_metrics, honeypot, retry_count, status, assigned_to, created_at, updated_at
        ) VALUES (
            %(task_id)s, %(image_id)s, %(image_url)s, %(bounding_box)s, %(baseline_mask)s, %(queue)s,
            %(routing_metrics)s, %(honeypot)s, %(retry_count)s, %(status)s, %(assigned_to)s, %(created_at)s, %(updated_at)s
        )
    """, {
        'task_id': task['task_id'],
        'image_id': task['image_id'],
        'image_url': task['image_url'],
        'bounding_box': json.dumps(task['bounding_box']),
        'baseline_mask': json.dumps(task['baseline_mask']),
        'queue': task['queue'],
        'routing_metrics': json.dumps(task['routing_metrics']),
        'honeypot': json.dumps(task['honeypot']),
        'retry_count': task.get('retry_count', 0),
        'status': task.get('status', 'pending'),
        'assigned_to': task.get('assigned_to'),
        'created_at': task.get('created_at'),
        'updated_at': task.get('updated_at')
    })
    
    conn.commit()
    cursor.close()
    conn.close()

def get_next_task(queue_name: str, annotator_id: str):
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    
    # Atomic claim
    cursor.execute(f"""
        UPDATE {queue_name}
        SET status = 'assigned', assigned_to = %s, updated_at = NOW()
        WHERE task_id = (
            SELECT task_id FROM {queue_name} 
            WHERE status = 'pending' 
            ORDER BY created_at ASC 
            FOR UPDATE SKIP LOCKED 
            LIMIT 1
        )
        RETURNING *;
    """, (annotator_id,))
    
    row = cursor.fetchone()
    conn.commit()
    cursor.close()
    conn.close()
    return row

def requeue_task_db(task_id: str):
    # Simplified: finding task across queues and requeuing
    pass # to be implemented
