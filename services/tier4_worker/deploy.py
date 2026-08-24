import logging
import uuid
import re

logger = logging.getLogger(__name__)

class BlueGreenDeployer:
    """
    Handles packaging the updated LoRA weights and hot-swapping 
    the inference containers via Traefik/Nginx routing.
    """
    
    def __init__(self):
        self.deployment_target = "green"  # Starts by deploying to green container
    
    def deploy_weights(self, old_version: str) -> str:
        """
        Mocks the deployment of the new weights and returns the new model_version.
        """
        # Parse the current version to increment it
        # e.g., "serving-ui-stochastic-0.1.0" -> "serving-ui-stochastic-0.1.1"
        match = re.search(r'(.*?)-(\d+\.\d+\.)(\d+)', old_version)
        if match:
            prefix = match.group(1)
            major_minor = match.group(2)
            patch = int(match.group(3))
            new_version = f"{prefix}-{major_minor}{patch + 1}"
        else:
            # Fallback
            new_version = f"{old_version}-rev-{uuid.uuid4().hex[:4]}"
            
        logger.info(f"Packaging new LoRA weights (.safetensors) for {new_version}...")
        logger.info(f"Loading weights into idle '{self.deployment_target}' inference container...")
        logger.info(f"Flipping Nginx routing to '{self.deployment_target}' container...")
        
        # Toggle target for next time
        self.deployment_target = "blue" if self.deployment_target == "green" else "green"
        
        logger.info(f"Deployment complete. New serving version is {new_version}")
        return new_version
