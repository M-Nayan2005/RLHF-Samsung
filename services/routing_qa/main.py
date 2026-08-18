from fastapi import FastAPI, HTTPException, Depends
from contextlib import asynccontextmanager
from poller import PollerThread
from db import get_next_task, get_db_connection

poller_thread = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global poller_thread
    poller_thread = PollerThread(interval=5)
    poller_thread.start()
    yield
    if poller_thread:
        poller_thread.stop()
        poller_thread.join()

app = FastAPI(title="Routing QA Service", lifespan=lifespan)

@app.get("/tasks/next")
async def get_next_task_endpoint(queue: str, annotator_id: str):
    allowed_queues = ["junior_queue", "senior_queue", "consensus_queue"]
    if queue not in allowed_queues:
        raise HTTPException(status_code=400, detail="Invalid queue")
        
    task = get_next_task(queue, annotator_id)
    if not task:
        return {"task": None, "message": "No pending tasks available"}
    
    # Strip ground_truth_mask for honeypots before sending to UI!
    task_dict = dict(task)
    if task_dict.get("honeypot") and task_dict["honeypot"].get("is_honeypot"):
        task_dict["honeypot"]["ground_truth_mask"] = None
        
    return {"task": task_dict}

@app.post("/tasks/{task_id}/requeue")
async def requeue_task(task_id: str):
    # Retrieve task, check retry count, if > MAX move to discard bin
    # We will implement a simplified stub for this
    return {"status": "requeued or discarded", "task_id": task_id}

@app.post("/tasks/{task_id}/honeypot-result")
async def honeypot_result(task_id: str):
    # Compare submitted mask vs ground truth and update trust score
    return {"status": "trust_score_updated", "task_id": task_id}
