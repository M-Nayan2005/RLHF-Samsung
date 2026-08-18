import os
import torch
import numpy as np
import httpx
import cv2
from io import BytesIO
from typing import Tuple, List, Dict, Any

# Grounding DINO
try:
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
except ImportError:
    print("Warning: groundingdino not installed.")

# SAM 2
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Warning: sam2 not installed.")


DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

try:
    import groundingdino
    default_dino_config = os.path.join(os.path.dirname(groundingdino.__file__), "config/GroundingDINO_SwinT_OGC.py")
except ImportError:
    default_dino_config = "groundingdino/config/GroundingDINO_SwinT_OGC.py"

GROUNDING_DINO_CONFIG = os.environ.get("GROUNDING_DINO_CONFIG", default_dino_config)
GROUNDING_DINO_CHECKPOINT = os.environ.get("GROUNDING_DINO_CHECKPOINT", "/weights/groundingdino_swint_ogc.pth")

SAM2_CONFIG = os.environ.get("SAM2_CONFIG", "sam2_hiera_large.yaml")
SAM2_CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "/weights/sam2_hiera_large.pt")

class InferencePipeline:
    def __init__(self):
        self.device = DEVICE
        self.dino_model = None
        self.sam2_predictor = None

    def load_models(self):
        if self.dino_model is None:
            self.dino_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT)
            self.dino_model = self.dino_model.to(self.device)
            self.dino_model.eval()

        if self.sam2_predictor is None:
            sam2_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=self.device)
            self.sam2_predictor = SAM2ImagePredictor(sam2_model)

    def _enable_mcd_in_sam2(self):
        """
        Enable Monte Carlo Dropout by keeping dropout layers in train mode during inference.
        """
        for m in self.sam2_predictor.model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    async def download_image(self, url: str) -> np.ndarray:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            image_bytes = np.frombuffer(response.content, np.uint8)
            image_bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            return image_rgb

    def transform_image_for_dino(self, image_rgb: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        # grounding dino expects PIL image
        from PIL import Image
        pil_image = Image.fromarray(image_rgb)
        image_transformed, _ = transform(pil_image, None)
        return image_transformed, np.asarray(pil_image)

    def run_grounding_dino(self, image_rgb: np.ndarray, text_prompt: str) -> Dict[str, Any]:
        image_transformed, _ = self.transform_image_for_dino(image_rgb)
        
        boxes, logits, phrases = predict(
            model=self.dino_model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=0.3,
            text_threshold=0.25,
            device=self.device
        )
        
        if len(boxes) == 0:
            raise ValueError(f"No objects found for prompt: {text_prompt}")
            
        # Get the highest confidence box
        best_idx = logits.argmax().item()
        box = boxes[best_idx].cpu().numpy()
        confidence = logits[best_idx].item()
        label = phrases[best_idx]
        
        # Grounding DINO outputs normalized [cx, cy, w, h]
        h, w, _ = image_rgb.shape
        cx, cy, bw, bh = box
        
        x_min = (cx - bw / 2) * w
        y_min = (cy - bh / 2) * h
        x_max = (cx + bw / 2) * w
        y_max = (cy + bh / 2) * h
        
        return {
            "x_min": float(x_min),
            "y_min": float(y_min),
            "x_max": float(x_max),
            "y_max": float(y_max),
            "label": label,
            "confidence": float(confidence)
        }

    def _convert_mask_to_polygon(self, mask: np.ndarray) -> List[List[float]]:
        # mask is boolean array
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
            
        # get the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        # Simplify the contour slightly
        epsilon = 0.005 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        polygon = []
        for point in approx:
            polygon.append([float(point[0][0]), float(point[0][1])])
        return polygon

    def run_sam2_mcd(self, image_rgb: np.ndarray, bbox: Dict[str, float], passes: int = 5) -> List[Dict[str, Any]]:
        self.sam2_predictor.set_image(image_rgb)
        self._enable_mcd_in_sam2()
        
        input_box = np.array([
            bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]
        ])
        
        mcd_samples = []
        for i in range(passes):
            # SAM 2 predicts multiple masks per prompt (usually 3). 
            # We take the best one per MCD pass.
            masks, scores, logits = self.sam2_predictor.predict(
                box=input_box,
                multimask_output=False
            )
            
            best_mask = masks[0]
            mask_logits = logits[0]
            
            # mask_logits could be dense logits, we can pool them or take a mean for class logits.
            # Usually, SAM gives a confidence score. If we strictly need per-class logits for entropy,
            # we might use a proxy if SAM doesn't output true categorical logits. 
            # We'll mock class logits using the SAM score and a random normal dist centered around it.
            # In a true multi-class segmentation, logits would be a vector. SAM is class-agnostic.
            # The prompt asks for "class_logit_entropy". We'll treat the foreground/background logits as the class logits.
            
            class_logits = [float(scores[0]), 1.0 - float(scores[0])] 
            
            polygon = self._convert_mask_to_polygon(best_mask)
            
            mcd_samples.append({
                "sample_index": i,
                "mask": {"points": polygon, "rle": None},
                "class_logits": class_logits
            })
            
        return mcd_samples

pipeline = InferencePipeline()
