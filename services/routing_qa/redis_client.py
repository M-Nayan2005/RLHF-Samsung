import redis
from config import settings

# Global redis connection pool
pool = redis.ConnectionPool.from_url(settings.REDIS_URL, decode_responses=True)

def get_redis():
    return redis.Redis(connection_pool=pool)

def push_to_redis_queue(queue_type: str, task_id: str):
    r = get_redis()
    
    redis_key = None
    if queue_type == "junior_queue":
        redis_key = settings.REDIS_JUNIOR_QUEUE_KEY
    elif queue_type == "senior_queue":
        redis_key = settings.REDIS_SENIOR_QUEUE_KEY
    elif queue_type == "consensus_queue":
        redis_key = settings.REDIS_CONSENSUS_QUEUE_KEY
        
    if redis_key:
        r.lpush(redis_key, task_id)
