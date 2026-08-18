from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timezone
import sys
import os

# Add common.schemas to path if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from common.schemas.tier1_ingestion import GroundedSAM2Output, BoundingBox, MCDSample, PolygonMask
from inference import pipeline
from metrics import calculate_geometric_variance_and_consensus, calculate_class_logit_entropy
from database import init_db, AsyncSessionLocal, Tier1Prediction

app = FastAPI(title="Pre-Inference Service")

class PredictRequest(BaseModel):
    image_url: str
    text_prompt: str

@app.on_event("startup")
async def startup_event():
    # Initialize Database
    await init_db()
    # Load Models
    pipeline.load_models()

@app.post("/predict", response_model=GroundedSAM2Output)
async def predict_endpoint(request: PredictRequest):
    try:
        # 1. Stream Image
        image_rgb = await pipeline.download_image(request.image_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {e}")

    try:
        # 2. Run Grounding DINO
        bbox_dict = pipeline.run_grounding_dino(image_rgb, request.text_prompt)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grounding DINO inference failed: {e}")

    try:
        # 3. Run SAM2 with 5x MCD
        mcd_samples_dicts = pipeline.run_sam2_mcd(image_rgb, bbox_dict, passes=5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SAM2 inference failed: {e}")

    # Extract polygon coordinates for geometric variance calculation
    polygons = [s["mask"]["points"] for s in mcd_samples_dicts]
    
    # Extract class logits for entropy
    logits_list = [s["class_logits"] for s in mcd_samples_dicts]

    # 4. Compute Metrics
    geometric_variance, consensus_polygon = calculate_geometric_variance_and_consensus(polygons)
    class_logit_entropy = calculate_class_logit_entropy(logits_list)

    # 5. Construct Response
    image_id = f"img_{uuid.uuid4().hex[:8]}"
    
    output = GroundedSAM2Output(
        image_id=image_id,
        image_url=request.image_url,
        text_prompt=request.text_prompt,
        bounding_box=BoundingBox(**bbox_dict),
        mcd_samples=[MCDSample(**s) for s in mcd_samples_dicts],
        geometric_variance=geometric_variance,
        class_logit_entropy=class_logit_entropy,
        consensus_mask=PolygonMask(points=consensus_polygon),
        model_version="grounding-dino-1.0+sam2-hiera-l",
        created_at=datetime.now(timezone.utc).isoformat()
    )

    # 6. Persist to Postgres
    async with AsyncSessionLocal() as session:
        db_record = Tier1Prediction(
            image_id=output.image_id,
            image_url=output.image_url,
            text_prompt=output.text_prompt,
            bounding_box=output.bounding_box.model_dump(),
            mcd_samples=[s.model_dump() for s in output.mcd_samples],
            geometric_variance=output.geometric_variance,
            class_logit_entropy=output.class_logit_entropy,
            consensus_mask=output.consensus_mask.model_dump(),
            model_version=output.model_version,
            created_at=datetime.fromisoformat(output.created_at).replace(tzinfo=None)
        )
        session.add(db_record)
        await session.commit()

    return output

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
