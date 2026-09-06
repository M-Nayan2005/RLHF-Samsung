# Tier 4 Implementation Details
## PPO Trainer and LoRA Hot-Swapping

As specified, this Tier handles the offline training loop. The following components were implemented:

1. **Sampler (`services/tier4_worker/dataset.py`)**: 
   - Uses `asyncpg` to query the `tier3.replay_buffer` table.
   - Samples randomized training batches of exactly 64 tuples matching the current model version.
   
2. **Trainer (`services/tier4_worker/ppo.py`)**:
   - Implements Proximal Policy Optimization (PPO) using a Critic network to calculate advantages (`A_t = R_t - V(S_t)`).
   - Applies PPO policy gradient updates *exclusively* to the Low-Rank Adaptation (LoRA) weights of the SAM2 backbone to prevent catastrophic forgetting.
   
3. **Deployer (`services/tier4_worker/deploy.py`)**:
   - Handles the Blue/Green hot-swapping logic.
   - Packages and saves updated LoRA weights (`.safetensors`) to the shared `/app/models` volume.
   - Publishes a `RELOAD_LORA_WEIGHTS` signal via Redis Pub/Sub (`tier1_model_updates` channel) instructing the Tier 1 Inference engine to reload weights seamlessly.

4. **Integration**:
   - Appended `tier4_worker` container definition into `docker-compose.yml` with strict isolation.
   - Maintained all boundary rules: zero deletions and no modifications to Tier 1, Tier 2, or Tier 3 logic.
