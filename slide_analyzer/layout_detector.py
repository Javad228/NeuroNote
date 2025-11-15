"""Layout-aware detector using LayoutLMv3 (or similar) for diagram elements."""

from functools import lru_cache
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForObjectDetection

from .config import (
    LAYOUT_DETECTOR_MODEL,
    LAYOUT_DETECTOR_THRESHOLD,
    DEVICE,
)


@lru_cache(maxsize=1)
def _load_layout_detector():
    processor = AutoProcessor.from_pretrained(LAYOUT_DETECTOR_MODEL)
    model = AutoModelForObjectDetection.from_pretrained(LAYOUT_DETECTOR_MODEL).to(DEVICE)
    model.eval()
    return processor, model


def detect_layout_elements(image_pil: Image.Image) -> List[Dict[str, Any]]:
    """Detect layout/text elements to use as fallback anchors."""
    try:
        processor, model = _load_layout_detector()
    except Exception:
        return []

    inputs = processor(images=image_pil, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image_pil.size[::-1]], device=DEVICE)
    results = processor.post_process_object_detection(
        outputs, threshold=LAYOUT_DETECTOR_THRESHOLD, target_sizes=target_sizes
    )[0]

    elements: List[Dict[str, Any]] = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        elements.append({
            "label": model.config.id2label.get(label.item(), f"class_{label.item()}"),
            "score": float(score.item()),
            "box": [float(x) for x in box.cpu().tolist()],
        })
    return elements

