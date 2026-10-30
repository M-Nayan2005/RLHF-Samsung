"""
Publish a trained policy so the next served task uses it.

Serving (`services/serving_ui`) reads `<models_dir>/policy/current.json` on
every request, so publishing is: write the new checkpoint, then atomically
flip the pointer. That is the blue/green switch - the old checkpoint stays on
disk, untouched, and rolling back is pointing `current.json` at it again.

A Redis message on `policy_updates` is also sent, for anything that wants to
react to a deploy; nothing depends on it being received.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from common import rl_policy

logger = logging.getLogger(__name__)

UPDATES_CHANNEL = "policy_updates"


class PolicyPublisher:
    def __init__(self, models_dir: str, default_version: str, redis_url: Optional[str] = None):
        self.store = rl_policy.PolicyStore(models_dir, default_version)
        self._redis = None
        if redis_url:
            try:
                import redis

                self._redis = redis.Redis.from_url(redis_url, socket_timeout=2.0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("No Redis for deploy notifications (%s); publishing to disk only", exc)

    def critic_path(self, version: str) -> Path:
        return self.store.path_for(version).with_suffix(".critic.pt")

    def load_critic(self, version: str) -> Optional[Dict[str, Any]]:
        path = self.critic_path(version)
        if not path.exists():
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def publish(self, params: rl_policy.PolicyParams, critic_state: Dict[str, Any]) -> Path:
        # Critic first: once the pointer flips, the next training run looks for it.
        torch.save(critic_state, self.critic_path(params.version))
        path = self.store.save(params, make_current=True)
        logger.info("Published policy %s (parent %s) -> %s", params.version, params.parent_version, path)

        if self._redis is not None:
            try:
                self._redis.publish(UPDATES_CHANNEL, json.dumps({
                    "action": "POLICY_PUBLISHED",
                    "model_version": params.version,
                    "parent_version": params.parent_version,
                    "path": str(path),
                }))
            except Exception as exc:  # noqa: BLE001 - the pointer file is the source of truth
                logger.warning("Could not publish the deploy notification: %s", exc)
        return path
