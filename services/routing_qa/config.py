import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/rlhf_seg")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    REDIS_JUNIOR_QUEUE_KEY: str = os.getenv("REDIS_JUNIOR_QUEUE_KEY", "queue:junior")
    REDIS_SENIOR_QUEUE_KEY: str = os.getenv("REDIS_SENIOR_QUEUE_KEY", "queue:senior")
    REDIS_CONSENSUS_QUEUE_KEY: str = os.getenv("REDIS_CONSENSUS_QUEUE_KEY", "queue:consensus")
    
    STOCHASTIC_AUDIT_RATE: float = float(os.getenv("STOCHASTIC_AUDIT_RATE", "0.05"))
    HONEYPOT_INJECTION_RATE: float = float(os.getenv("HONEYPOT_INJECTION_RATE", "0.10"))
    MAX_CONSENSUS_RETRIES: int = int(os.getenv("MAX_CONSENSUS_RETRIES", "3"))
    
    # Thresholds for routing (can be overwritten via env)
    VARIANCE_THRESHOLD: float = float(os.getenv("VARIANCE_THRESHOLD", "0.05"))
    ENTROPY_THRESHOLD: float = float(os.getenv("ENTROPY_THRESHOLD", "0.20"))

settings = Settings()
