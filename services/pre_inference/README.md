# Pre-Inference & Auto-Labeling Engine

This service takes an image URL and a text prompt, runs Grounding DINO to find the bounding box, then runs SAM2 5 times with Monte Carlo Dropout (MCD) to generate 5 variant polygon masks. It computes uncertainty metrics and persists the outputs to PostgreSQL.

## Architecture

*   **FastAPI**: Provides the asynchronous entrypoint.
*   **Database**: PostgreSQL using `asyncpg` and SQLAlchemy 2.0 to ensure a non-blocking event loop.
*   **Deployment**: Modal Labs. This serverless containerization solves the Cold Start problem through image caching and handles Cloud Secrets securely.
*   **Metrics**: 
    *   `geometric_variance`: Mean pairwise Intersection over Union (IoU) distance.
    *   `class_logit_entropy`: Shannon entropy of the softmax-averaged class logits across MCD samples.
    *   `consensus_mask`: Medoid polygon (the polygon with highest average IoU to other polygons).

## Deployment

Deploy to Modal Labs:
```bash
modal deploy modal_app.py
```
