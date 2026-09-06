import logging
import uuid
import re
import os
import json
import redis

logger = logging.getLogger(__name__)

class BlueGreenDeployer:
    """
    Handles packaging the updated LoRA weights, saving them to the shared volume,
    and hot-swapping the inference containers via a Redis Pub/Sub signal.
    """
    
    def __init__(self, models_dir: str = "/app/models", redis_url: str = "redis://localhost:6379/0"):
        self.models_dir = models_dir
        self.deployment_target = "green"  
        
        # Connect to Redis
        try:
            self.redis_client = redis.Redis.from_url(redis_url)
        except Exception as e:
            logger.warning(f"Could not connect to Redis at {redis_url}: {e}")
            self.redis_client = None
            
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"Created models directory at {self.models_dir}")
    
    def deploy_weights(self, old_version: str) -> str:
        """
        Saves the new LoRA weights to the shared volume and signals Tier 1.
        """
        match = re.search(r'(.*?)-(\d+\.\d+\.)(\d+)', old_version)
        if match:
            prefix = match.group(1)
            major_minor = match.group(2)
            patch = int(match.group(3))
            new_version = f"{prefix}-{major_minor}{patch + 1}"
        else:
            new_version = f"{old_version}-rev-{uuid.uuid4().hex[:4]}"
            
        logger.info(f"Packaging new LoRA weights (.safetensors) for {new_version}...")
        
        # Save weights to the shared volume
        weight_path = os.path.join(self.models_dir, f"{new_version}_lora.safetensors")
        try:
            with open(weight_path, "w") as f:
                f.write("mock_safetensors_content_with_updated_lora_weights")
            logger.info(f"Saved new weights to {weight_path}")
        except Exception as e:
            logger.error(f"Failed to write weights to {weight_path}: {e}")
        
        # Signal Tier 1 via Redis Pub/Sub to seamlessly reload weights
        payload = {
            "action": "RELOAD_LORA_WEIGHTS",
            "model_version": new_version,
            "weight_path": weight_path,
            "target_container": self.deployment_target
        }
        
        if self.redis_client:
            try:
                self.redis_client.publish("tier1_model_updates", json.dumps(payload))
                logger.info(f"Published Redis signal to 'tier1_model_updates': {payload}")
            except Exception as e:
                logger.error(f"Failed to publish to Redis: {e}")
        else:
            logger.warning("Redis client not initialized; skipping Pub/Sub signal")
        
        # Toggle target for next time
        self.deployment_target = "blue" if self.deployment_target == "green" else "green"
        
        logger.info(f"Deployment complete. New serving version is {new_version}")
        return new_version
