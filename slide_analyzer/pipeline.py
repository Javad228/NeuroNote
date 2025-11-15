"""Main pipeline orchestration for slide analysis."""

from pathlib import Path
import json
import numpy as np

from .config import DEFAULT_OUTPUT_DIR
from .image_utils import load_image_from_path_or_b64, pil_to_base64_png, normalize_xywh_to_xyxy
from .gpt5_vision import setup_openai_client, generate_explanation_with_gpt5
from .grounding_dino import load_grounding_dino, refine_boxes_with_gdn
from .sam_segmentation import load_sam_model, box_to_best_mask
from .visualization import draw_bounding_boxes, get_mask_bounding_box
from .ocr_utils import extract_text_regions
from .layout_detector import detect_layout_elements
from .image_utils import rect_mask


def _gather_detection_phrases(sentence: dict) -> list:
    phrases = sentence.get("detection_phrases")
    if isinstance(phrases, str):
        phrases = [p.strip() for p in phrases.split(",") if p.strip()]
    elif isinstance(phrases, list):
        phrases = [str(p).strip() for p in phrases if str(p).strip()]
    else:
        phrases = []
    if not phrases:
        text = sentence.get("text", "")
        phrases = [text[:32]] if text else []
    return phrases


def _match_ocr_box(sentence: dict, ocr_results):
    phrases = _gather_detection_phrases(sentence)
    phrases = [p.lower() for p in phrases]
    for anchor in ocr_results:
        text = anchor.get("text", "").lower()
        if not text:
            continue
        for phrase in phrases:
            if text in phrase or phrase in text:
                return anchor.get("box")
    return None


def analyze_slide(image_path_or_b64, openai_api_key=None, 
                 output_dir=DEFAULT_OUTPUT_DIR):
    """
    Main pipeline function to analyze a slide and generate intermediate visualizations.
    
    Creates separate images for each sentence at each step:
    - GPT-5 coarse bounding boxes (one image per sentence)
    - Grounding DINO refined boxes (one image per sentence)
    - SAM mask bounding boxes (one image per sentence)
    
    Args:
        image_path_or_b64: Path to image file or base64 string
        openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        output_dir: Directory to save output images
    
    Returns:
        dict: Results with intermediate images and data
    """
    print("\n" + "=" * 60)
    print("GPT-5 SLIDE ANALYZER")
    print("=" * 60)
    
    # Setup
    print("\nLoading models...")
    client = setup_openai_client(openai_api_key)
    processor, gdn = load_grounding_dino()
    sam_pred = load_sam_model()
    
    # Load image
    print(f"\nLoading image...")
    im = load_image_from_path_or_b64(image_path_or_b64)
    print(f"✓ Image loaded: {im.size[0]}×{im.size[1]} pixels")
    
    # Extract OCR anchors for real text labels
    print("\nRunning OCR for real anchors...")
    ocr_results = extract_text_regions(im)
    ocr_snippets = [item["text"] for item in ocr_results]
    print(f"  ✓ OCR found {len(ocr_results)} text snippets")

    # Save OCR results
    ocr_path = Path(output_dir) / "ocr_results.json"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    with open(ocr_path, "w", encoding="utf-8") as f:
        import json
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ OCR annotations: {ocr_path}")

    # Detect layout elements (text boxes, arrows, etc.)
    print("\nRunning layout-aware detector...")
    layout_elements = detect_layout_elements(im)
    layout_path = Path(output_dir) / "layout_detector_results.json"
    with open(layout_path, "w", encoding="utf-8") as f:
        json.dump(layout_elements, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Layout detections: {len(layout_elements)} (saved to {layout_path})")

    # Get GPT-5 analysis
    b64_png = pil_to_base64_png(im)
    sentences, raw_gpt5_output = generate_explanation_with_gpt5(
        b64_png, client, ocr_snippets=ocr_snippets
    )
    
    print(f"\nGPT-5 generated {len(sentences)} sentences")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s['text'][:80]}...")
        phrases = s.get("detection_phrases") or []
        if isinstance(phrases, str):
            phrases = [phrases]
        phrase_str = ", ".join(phrases) if phrases else s.get("keywords", "N/A")
        print(f"      Detection phrases: {phrase_str}")
    
    # Save GPT-5 raw output
    print(f"\nSaving outputs to '{output_dir}'...")
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    gpt5_output_path = Path(output_dir) / "gpt5_raw_output.txt"
    with open(gpt5_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_gpt5_output)
    print(f"  ✓ GPT-5 raw output: {gpt5_output_path}")
    
    # Create output directories for each step
    gpt5_dir = Path(output_dir) / "01_gpt5_coarse_boxes"
    dino_dir = Path(output_dir) / "02_grounding_dino_boxes"
    sam_dir = Path(output_dir) / "03_sam_boxes"
    
    gpt5_dir.mkdir(exist_ok=True)
    dino_dir.mkdir(exist_ok=True)
    sam_dir.mkdir(exist_ok=True)
    
    W, H = im.size
    
    # Step 1: Visualize GPT-5 coarse bounding boxes (one per sentence)
    print("\nStep 1: Visualizing GPT-5 coarse bounding boxes...")
    gpt5_images = []
    for idx, s in enumerate(sentences, 1):
        x1, y1, x2, y2 = normalize_xywh_to_xyxy(
            s["x"], s["y"], s["width"], s["height"], W, H
        )
        box = [x1, y1, x2, y2]
        viz = draw_bounding_boxes(im, [box], labels=[f"{idx}"])
        
        output_path = gpt5_dir / f"sentence_{idx:02d}.png"
        viz.save(output_path)
        gpt5_images.append(viz)
        print(f"  ✓ Sentence {idx}: {output_path}")
    
    # Step 2: Refine with Grounding DINO
    print("\nStep 2: Refining with Grounding DINO...")
    refined = refine_boxes_with_gdn(
        sentences, im, processor, gdn,
        ocr_results=ocr_results,
        layout_elements=layout_elements
    )
    
    # Visualize Grounding DINO bounding boxes (one per sentence)
    print("  Visualizing Grounding DINO bounding boxes...")
    dino_images = []
    for idx, (s, xyxy, score) in enumerate(refined, 1):
        label = f"{idx} ({score:.2f})" if score > 0 else f"{idx} (GPT-5)"
        viz = draw_bounding_boxes(im, [xyxy], labels=[label])
        
        output_path = dino_dir / f"sentence_{idx:02d}.png"
        viz.save(output_path)
        dino_images.append(viz)
        print(f"  ✓ Sentence {idx}: {output_path}")
    
    # Step 3: Get SAM masks and visualize bounding boxes (one per sentence)
    print("\nStep 3: Generating SAM masks...")
    sam_images = []
    sam_boxes_list = []
    
    for idx, (s, xyxy, score) in enumerate(refined, 1):
        mask_bool = box_to_best_mask(im, xyxy, sam_pred)
        sam_box = get_mask_bounding_box(mask_bool)

        if sam_box and np.sum(mask_bool) > 0:
            sam_boxes_list.append(sam_box)
            label = f"{idx}"
        else:
            ocr_box = _match_ocr_box(s, ocr_results)
            if ocr_box:
                sam_boxes_list.append(ocr_box)
                label = f"{idx} (ocr)"
            else:
                sam_box = xyxy
                sam_boxes_list.append(sam_box)
                label = f"{idx} (fallback)"
        
        viz = draw_bounding_boxes(im, [sam_box], labels=[label])
        
        output_path = sam_dir / f"sentence_{idx:02d}.png"
        viz.save(output_path)
        sam_images.append(viz)
        print(f"  ✓ Sentence {idx}: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  • {gpt5_output_path}")
    print(f"  • {gpt5_dir}/ ({(len(sentences))} images)")
    print(f"  • {dino_dir}/ ({(len(refined))} images)")
    print(f"  • {sam_dir}/ ({(len(sam_images))} images)")
    
    return {
        "gpt5_raw_output": raw_gpt5_output,
        "sentences": sentences,
        "refined_items": refined,
        "gpt5_images": gpt5_images,
        "dino_images": dino_images,
        "sam_images": sam_images,
    }
