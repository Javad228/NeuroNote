"""OCR utilities for extracting real anchors from slides."""

from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from PIL import Image

from .config import OCR_MIN_CONFIDENCE, OCR_MAX_RESULTS, OCR_USE_GPU

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except Exception:
    _EASYOCR_AVAILABLE = False

try:
    import pytesseract
    from pytesseract import Output as TesseractOutput
    _PYTESS_AVAILABLE = True
except Exception:
    _PYTESS_AVAILABLE = False


def _normalize_bbox(bbox) -> List[int]:
    """Convert EasyOCR polygon bbox to [x1, y1, x2, y2]."""
    xs = [int(pt[0]) for pt in bbox]
    ys = [int(pt[1]) for pt in bbox]
    return [min(xs), min(ys), max(xs), max(ys)]


def _easyocr_extract(image_pil: Image.Image) -> List[Dict[str, Any]]:
    reader = easyocr.Reader(["en"], gpu=OCR_USE_GPU)
    results = reader.readtext(np.array(image_pil))
    anchors = []
    for bbox, text, conf in results:
        if conf < OCR_MIN_CONFIDENCE:
            continue
        anchors.append({
            "text": text.strip(),
            "confidence": float(conf),
            "box": _normalize_bbox(bbox),
            "source": "easyocr",
        })
    return anchors


def _pytesseract_extract(image_pil: Image.Image) -> List[Dict[str, Any]]:
    data = pytesseract.image_to_data(
        image_pil,
        output_type=TesseractOutput.DICT
    )
    anchors = []
    n = len(data["text"])
    for i in range(n):
        text = data["text"][i].strip()
        if not text:
            continue
        conf = float(data["conf"][i])
        if conf < (OCR_MIN_CONFIDENCE * 100):  # pytesseract conf is in [0,100]
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        anchors.append({
            "text": text,
            "confidence": conf / 100.0,
            "box": [x, y, x + w, y + h],
            "source": "pytesseract",
        })
    return anchors


def extract_text_regions(image_pil: Image.Image) -> List[Dict[str, Any]]:
    """Extract OCR anchors from the image (text + bounding boxes)."""
    anchors: List[Dict[str, Any]] = []

    if _EASYOCR_AVAILABLE:
        try:
            anchors = _easyocr_extract(image_pil)
        except Exception:
            anchors = []

    if not anchors and _PYTESS_AVAILABLE:
        try:
            anchors = _pytesseract_extract(image_pil)
        except Exception:
            anchors = []

    # Deduplicate and sort by confidence
    seen = set()
    unique = []
    for anchor in anchors:
        key = (anchor["text"].lower(), tuple(anchor["box"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(anchor)

    unique.sort(key=lambda a: a.get("confidence", 0.0), reverse=True)
    return unique[:OCR_MAX_RESULTS]

