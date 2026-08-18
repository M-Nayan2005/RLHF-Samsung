# Routing & QA Service

## Responsibilities
- Poll Postgres (`tier1_predictions`) for new auto-labeled segmentations.
- Apply dual-metric routing thresholds.
- Manage queue state in Postgres (`junior_queue`, `senior_queue`, `consensus_queue`).
- Mirror task IDs into Redis lists for fast downstream polling.
- Inject honeypots and manage stochastic audit filtering.

## Routing Logic & Thresholds
Developer 2 relies on the computed values received from Developer 1 (`geometric_variance` and `class_logit_entropy`) to route tasks downstream to Developer 3's UI.

The thresholds are configured as follows:
- **`VARIANCE_THRESHOLD`**: `0.05`
- **`ENTROPY_THRESHOLD`**: `0.20`

**Routing Rules:**
1. **Junior Queue**: Variance < 0.05 AND Entropy < 0.20
2. **Senior Queue**: Entropy >= 0.20 (confidently wrong) OR Variance >= 0.05 (messy mask)
3. **Consensus Queue**: Extreme outliers (e.g., Variance >= 0.10 AND Entropy >= 0.40) OR selected by the Stochastic Audit Filter.
