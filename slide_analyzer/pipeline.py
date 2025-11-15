"""Main pipeline orchestration for slide analysis using Florence-2 + SAM."""

from pathlib import Path
import json

from .config import DEFAULT_OUTPUT_DIR
from .image_utils import load_image_from_path_or_b64, pil_to_base64_png, normalize_xywh_to_xyxy
from .gpt5_vision import setup_openai_client, generate_explanation_with_gpt5
from .florence2_detector import load_florence2_detector
from .sam_segmentation import load_sam_model, box_to_best_mask
from .visualization import draw_bounding_boxes, get_mask_bounding_box


def analyze_slide(image_path_or_b64, openai_api_key=None, 
                 output_dir=DEFAULT_OUTPUT_DIR):
    """
    Main pipeline function to analyze a slide and generate intermediate visualizations.
    
    Pipeline: Florence-2 OCR → GPT-5 Vision → Florence-2 Phrase Grounding → SAM
    
    Args:
        image_path_or_b64: Path to image file or base64 string
        openai_api_key: OpenAI API key (uses OPENAI_API_KEY env var if None)
        output_dir: Directory to save output images
    
    Returns:
        dict: Results with intermediate images and data
    """
    print("\n" + "=" * 60)
    print("GPT-5 SLIDE ANALYZER (Florence-2 + SAM)")
    print("=" * 60)
    
    # Setup
    print("\nLoading models...")
    client = setup_openai_client(openai_api_key)
    florence = load_florence2_detector()
    sam_pred = load_sam_model()
    
    # Load image
    print(f"\nLoading image...")
    im = load_image_from_path_or_b64(image_path_or_b64)
    print(f"✓ Image loaded: {im.size[0]}×{im.size[1]} pixels")
    
    # Step 1: Extract OCR with Florence-2
    print("\nStep 1: Running Florence-2 OCR...")
    ocr_results = florence.extract_ocr_with_boxes(im)
    print(f"  ✓ Found {len(ocr_results)} text regions")
    
    # Save OCR results
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    ocr_path = Path(output_dir) / "florence2_ocr_results.json"
    with open(ocr_path, 'w', encoding='utf-8') as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)
    print(f"  ✓ OCR results: {ocr_path}")
    
    # Extract text snippets for GPT-5
    ocr_texts = [r['text'] for r in ocr_results if r['text'].strip()]
    
    # Step 2: Get GPT-5 analysis with OCR context
    print("\nStep 2: GPT-5 Vision analysis with OCR context...")
    b64_png = pil_to_base64_png(im)
    sentences, raw_gpt5_output = generate_explanation_with_gpt5(b64_png, client, ocr_texts)
    
    print(f"\nGPT-5 generated {len(sentences)} sentences")
    for i, s in enumerate(sentences, 1):
        print(f"  {i}. {s['text'][:80]}...")
        phrases = s.get('detection_phrases', [])
        if isinstance(phrases, list):
            print(f"      Detection phrases: {', '.join(phrases)}")
        else:
            print(f"      Detection phrases: {phrases}")
    
    # Save GPT-5 output
    gpt5_output_path = Path(output_dir) / "gpt5_raw_output.txt"
    with open(gpt5_output_path, 'w', encoding='utf-8') as f:
        f.write(raw_gpt5_output)
    print(f"\n  ✓ GPT-5 raw output: {gpt5_output_path}")
    
    # Create output directories
    gpt5_dir = Path(output_dir) / "01_gpt5_coarse_boxes"
    florence_dir = Path(output_dir) / "02_florence2_grounded_boxes"
    sam_dir = Path(output_dir) / "03_sam_boxes"
    
    gpt5_dir.mkdir(exist_ok=True)
    florence_dir.mkdir(exist_ok=True)
    sam_dir.mkdir(exist_ok=True)
    
    W, H = im.size
    
    # Step 3: Visualize GPT-5 coarse bounding boxes
    print("\nStep 3: Visualizing GPT-5 coarse bounding boxes...")
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
    
    # Step 4: Refine with Florence-2 phrase grounding
    print("\nStep 4: Refining with Florence-2 phrase grounding...")
    refined_items = []
    
    for idx, s in enumerate(sentences, 1):
        phrases = s.get('detection_phrases', [])
        if isinstance(phrases, str):
            phrases = [p.strip() for p in phrases.split(',')]
        elif not isinstance(phrases, list):
            phrases = [str(phrases)]
        
        print(f"  Sentence {idx}: trying detection phrases -> {phrases}")
        
        detected_box = None
        successful_phrase = None
        
        # Try each phrase with Florence-2
        for phrase in phrases:
            if not phrase or not phrase.strip():
                continue
            
            phrase = phrase.strip()
            print(f"    • Trying phrase: '{phrase}'")
            
            try:
                boxes = florence.phrase_grounding(im, phrase)
                if boxes and len(boxes) > 0:
                    detected_box = boxes[0]  # Take first match
                    successful_phrase = phrase
                    print(f"      ✓ Detected with phrase '{phrase}'")
                    break
            except Exception as e:
                print(f"      → Florence-2 error: {e}")
                continue
        
        if detected_box:
            refined_items.append((s, detected_box, 1.0, successful_phrase))
            print(f"      ✓ Using detection from phrase '{successful_phrase}'")
        else:
            # Fallback to GPT-5 box
            x1, y1, x2, y2 = normalize_xywh_to_xyxy(
                s["x"], s["y"], s["width"], s["height"], W, H
            )
            refined_items.append((s, [x1, y1, x2, y2], 0.0, None))
            print(f"      → No detections, using GPT-5 box")
    
    print(f"✓ Refined {len(refined_items)} regions")
    
    # Visualize Florence-2 results
    print("  Visualizing Florence-2 grounded boxes...")
    florence_images = []
    for idx, (s, xyxy, score, phrase) in enumerate(refined_items, 1):
        label = f"{idx} ({phrase[:20]}...)" if phrase else f"{idx} (GPT-5)"
        viz = draw_bounding_boxes(im, [xyxy], labels=[label])
        
        output_path = florence_dir / f"sentence_{idx:02d}.png"
        viz.save(output_path)
        florence_images.append(viz)
        print(f"  ✓ Sentence {idx}: {output_path}")
    
    # Step 5: Generate SAM masks
    print("\nStep 5: Generating SAM masks...")
    sam_images = []
    sam_boxes_list = []
    
    for idx, (s, xyxy, score, phrase) in enumerate(refined_items, 1):
        mask_bool = box_to_best_mask(im, xyxy, sam_pred)
        sam_box = get_mask_bounding_box(mask_bool)
        
        if sam_box:
            sam_boxes_list.append(sam_box)
            label = f"{idx}"
        else:
            # Fallback to original box if mask is empty
            sam_box = xyxy
            sam_boxes_list.append(sam_box)
            label = f"{idx} (empty)"
        
        viz = draw_bounding_boxes(im, [sam_box], labels=[label])
        
        output_path = sam_dir / f"sentence_{idx:02d}.png"
        viz.save(output_path)
        sam_images.append(viz)
        print(f"  ✓ Sentence {idx}: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  • {ocr_path}")
    print(f"  • {gpt5_output_path}")
    print(f"  • {gpt5_dir}/ ({len(sentences)} images)")
    print(f"  • {florence_dir}/ ({len(refined_items)} images)")
    print(f"  • {sam_dir}/ ({len(sam_images)} images)")
    
    return {
        "ocr_results": ocr_results,
        "gpt5_raw_output": raw_gpt5_output,
        "sentences": sentences,
        "refined_items": refined_items,
        "gpt5_images": gpt5_images,
        "florence_images": florence_images,
        "sam_images": sam_images,
    }
