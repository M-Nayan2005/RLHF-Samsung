import time
import threading
from db import get_unrouted_predictions, insert_queue_task
from router import build_queue_task
from redis_client import push_to_redis_queue

class PollerThread(threading.Thread):
    def __init__(self, interval: int = 5):
        super().__init__()
        self.interval = interval
        self._stop_event = threading.Event()
        self.daemon = True
        
    def stop(self):
        self._stop_event.set()
        
    def run(self):
        print("Starting routing poller thread...")
        while not self._stop_event.is_set():
            try:
                predictions = get_unrouted_predictions()
                for p in predictions:
                    # Route and wrap it
                    task = build_queue_task(p)
                    
                    # Insert to PG queue
                    insert_queue_task(task)
                    
                    # Mirror to Redis
                    push_to_redis_queue(task["queue"], task["task_id"])
                    
                    print(f"Routed {p['image_id']} -> {task['queue']} ({task['task_id']})")
                    
            except Exception as e:
                print(f"Poller error: {e}")
                
            time.sleep(self.interval)
        print("Routing poller thread stopped.")
