"""Segment Anything Model (SAM) integration for precise segmentation."""

import numpy as np
from pathlib import Path
from PIL import Image

from .config import (
    SAM_MODEL_TYPE,
    SAM_CHECKPOINT,
    SAM_CHECKPOINT_URL,
    SAM_HQ_CHECKPOINT,
    SAM_HQ_CHECKPOINT_URL,
    USE_SAM_HQ,
    DEVICE,
)

try:
    from segment_anything_hq import sam_model_registry as sam_hq_registry
    _SAM_HQ_AVAILABLE = True
except Exception:
    _SAM_HQ_AVAILABLE = False

from segment_anything import sam_model_registry, SamPredictor


def download_checkpoint(path: Path, url: str, label: str):
    if path.exists():
        return
    print(f"Downloading {label} checkpoint...")
    import urllib.request
    url = url.replace(" ", "%20")
    urllib.request.urlretrieve(url, path)
    print(f"✓ {label} checkpoint downloaded")


def load_sam_model():
    """Load SAM (or SAM-HQ) model and create predictor."""
    use_hq = USE_SAM_HQ and _SAM_HQ_AVAILABLE and Path(SAM_HQ_CHECKPOINT).exists()
    if USE_SAM_HQ and _SAM_HQ_AVAILABLE and not Path(SAM_HQ_CHECKPOINT).exists():
        download_checkpoint(Path(SAM_HQ_CHECKPOINT), SAM_HQ_CHECKPOINT_URL, "SAM-HQ")
        use_hq = True
    elif USE_SAM_HQ and not _SAM_HQ_AVAILABLE:
        print("⚠ SAM-HQ package not available, falling back to standard SAM.")

    if use_hq:
        print("Loading SAM-HQ model...")
        registry = sam_hq_registry
        checkpoint_path = SAM_HQ_CHECKPOINT
    else:
        print("Loading SAM model...")
        registry = sam_model_registry
        checkpoint_path = SAM_CHECKPOINT
        download_checkpoint(Path(SAM_CHECKPOINT), SAM_CHECKPOINT_URL, "SAM")

    sam = registry[SAM_MODEL_TYPE](checkpoint=checkpoint_path).to(DEVICE)
    sam_pred = SamPredictor(sam)
    print("✓ SAM/HQ model loaded")
    return sam_pred


def box_to_best_mask(image_pil: Image.Image, xyxy, sam_pred):
    """Convert bounding box to best segmentation mask using SAM."""
    img = np.array(image_pil)
    sam_pred.set_image(img)
    box = np.array(xyxy, dtype=np.float32)
    
    masks, scores, _ = sam_pred.predict(
        box=box[None, :], 
        multimask_output=True
    )
    
    best_idx = int(np.argmax(scores))
    return masks[best_idx].astype(bool)
