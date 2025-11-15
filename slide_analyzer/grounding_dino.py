"""Grounding DINO integration for bounding box refinement."""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

from .config import (
    GROUNDING_DINO_MODEL,
    DEVICE,
    DEFAULT_BOX_THRESHOLD,
    DEFAULT_TEXT_THRESHOLD,
)
from .image_utils import normalize_xywh_to_xyxy


def load_grounding_dino():
    """Load Grounding DINO model and processor."""
    print(f"Loading Grounding DINO...")
    processor = AutoProcessor.from_pretrained(GROUNDING_DINO_MODEL)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        GROUNDING_DINO_MODEL
    ).to(DEVICE)
    print("✓ Grounding DINO loaded")
    return processor, model


def _gdn_postprocess(outputs, inputs, image_pil, processor, 
                    box_threshold=DEFAULT_BOX_THRESHOLD, 
                    text_threshold=DEFAULT_TEXT_THRESHOLD):
    """Post-process Grounding DINO outputs, handling different transformers versions."""
    target_sizes = [image_pil.size[::-1]]
    
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        pass
    
    try:
        return processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError:
        pass
    
    try:
        return processor.post_process_object_detection(
            outputs,
            threshold=text_threshold,
            target_sizes=target_sizes,
        )[0]
    except TypeError as e:
        raise RuntimeError(f"Grounding DINO postprocess not supported: {e}")


def refine_with_gdn(image_pil: Image.Image, text_query: str, 
                   processor, model,
                   box_threshold=DEFAULT_BOX_THRESHOLD,
                   text_threshold=DEFAULT_TEXT_THRESHOLD):
    """Refine a single bounding box using Grounding DINO."""
    inputs = processor(images=image_pil, text=text_query, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    return _gdn_postprocess(
        outputs, inputs, image_pil, processor, 
        box_threshold, text_threshold
    )


def refine_boxes_with_gdn(sentences, image_pil: Image.Image, processor, model,
                         ocr_results=None, layout_elements=None):
    """Refine all sentence bounding boxes using Grounding DINO with GPT + OCR anchors."""
    W, H = image_pil.size
    refined_items = []
    
    print(f"\nRefining {len(sentences)} bounding boxes with Grounding DINO...")
    
    for idx, s in enumerate(sentences, 1):
        # Collect detection phrases (keywords optimized for object detection)
        raw_phrases = s.get("detection_phrases")
        if isinstance(raw_phrases, str):
            detection_phrases = [p.strip() for p in raw_phrases.split(",") if p.strip()]
        elif isinstance(raw_phrases, list):
            detection_phrases = [str(p).strip() for p in raw_phrases if str(p).strip()]
        else:
            detection_phrases = []

        if not detection_phrases:
            keywords = s.get("keywords", "")
            if isinstance(keywords, str):
                detection_phrases = [p.strip() for p in keywords.split(",") if p.strip()]

        if not detection_phrases:
            detection_phrases = [s.get("text", "")]

        # Extend with OCR text anchors
        if ocr_results:
            extra_phrases = []
            for anchor in ocr_results:
                txt = anchor.get("text", "").strip()
                if not txt:
                    continue
                extra_phrases.append(f"text '{txt}'")
            detection_phrases.extend(extra_phrases[:5])

        print(f"  Sentence {idx}: trying detection phrases -> {detection_phrases}")

        best_detection = None
        best_phrase = None

        for phrase in detection_phrases:
            if not phrase:
                continue
            query_text = phrase
            print(f"    • Trying phrase: '{query_text}'")
            try:
                result = refine_with_gdn(image_pil, query_text, processor, model)
                boxes = result.get("boxes", [])
                scores = result.get("scores", [])
            except Exception:
                boxes, scores = [], []

            if boxes is None or len(boxes) == 0:
                continue

            best_idx = int(torch.tensor(scores).argmax().item())
            refined_box = boxes[best_idx].cpu().numpy().tolist()
            confidence = float(scores[best_idx])
            best_detection = (refined_box, confidence)
            best_phrase = query_text
            print(f"      ✓ Detected with phrase '{query_text}' (confidence: {confidence:.3f})")
            break

        if best_detection is None and layout_elements:
            phrase_candidates = detection_phrases
            matched = _match_layout_box(phrase_candidates, layout_elements)
            if matched is not None:
                refined_items.append((s, matched, 0.2))
                print("      ✓ Using layout-detector fallback box")
                continue

        if best_detection is None:
            x1, y1, x2, y2 = normalize_xywh_to_xyxy(
                s["x"], s["y"], s["width"], s["height"], W, H
            )
            refined_items.append((s, [x1, y1, x2, y2], 0.0))
            print("      → No detections, using GPT-5 box")
        else:
            refined_box, confidence = best_detection
            refined_items.append((s, refined_box, confidence))
            print(f"      ✓ Using detection from phrase '{best_phrase}'")
    
    print(f"✓ Refined {len(refined_items)} regions")
    return refined_items


def _match_layout_box(phrases, layout_elements):
    """Find a fallback box from layout detector results based on phrase similarity."""
    if not layout_elements:
        return None
    phrases_lower = [p.lower() for p in phrases if isinstance(p, str)]
    for elem in layout_elements:
        label = elem.get("label", "").lower()
        if not label:
            continue
        for phrase in phrases_lower:
            if any(token in label for token in phrase.replace("'", "").split()):
                return elem.get("box")
    return None
