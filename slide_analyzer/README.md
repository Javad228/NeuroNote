# Slide Analyzer

Modular pipeline for analyzing educational slides using **Florence-2** + **GPT-5 Vision** + **SAM**.

## Pipeline Overview

```
Input Slide
    ↓
1. Florence-2 OCR (extract all text with bounding boxes)
    ↓
2. GPT-5 Vision (generate explanatory sentences + detection phrases using OCR context)
    ↓
3. Florence-2 Phrase Grounding (refine boxes using detection phrases)
    ↓
4. SAM (Segment Anything Model) (generate precise masks)
    ↓
Output: Visualizations at each step
```

## Why Florence-2?

Florence-2 is Microsoft's unified vision model specifically trained on:
- **Diagram understanding** (better than Grounding DINO for technical slides)
- **OCR with region detection** (built-in, no separate OCR needed)
- **Phrase grounding** (map text descriptions to image regions)
- **Dense captioning** (understand complex visual relationships)

## Installation

```bash
cd slide_analyzer
pip install -r requirements.txt
```

## Usage

### Command Line

```bash
python main.py input_slides/slide.png
```

### Python API

```python
from slide_analyzer import analyze_slide

result = analyze_slide("slide.png")

# Access results
ocr_results = result["ocr_results"]
sentences = result["sentences"]
final_image = result["florence_images"][-1]
```

## Output Structure

```
out/
├── florence2_ocr_results.json    # All detected text with boxes
├── gpt5_raw_output.txt           # Full GPT-5 response
├── 01_gpt5_coarse_boxes/         # GPT-5 initial boxes (one per sentence)
│   ├── sentence_01.png
│   └── sentence_02.png
├── 02_florence2_grounded_boxes/  # Florence-2 refined boxes (one per sentence)
│   ├── sentence_01.png
│   └── sentence_02.png
└── 03_sam_boxes/                 # SAM mask boxes (one per sentence)
    ├── sentence_01.png
    └── sentence_02.png
```

## Module Structure

```
slide_analyzer/
├── config.py              # Configuration constants
├── image_utils.py         # Image loading utilities
├── gpt5_vision.py         # GPT-5 Vision API integration
├── florence2_detector.py  # Florence-2 for OCR + grounding
├── sam_segmentation.py    # SAM for precise segmentation
├── visualization.py       # Bounding box drawing
├── pipeline.py            # Main orchestration
└── main.py                # CLI entry point
```

## Configuration

Edit `config.py` to customize:

```python
# Use Florence-2 base for faster inference (less accurate)
FLORENCE2_MODEL = "microsoft/Florence-2-base"

# Or use large for better accuracy (slower)
FLORENCE2_MODEL = "microsoft/Florence-2-large"

# Enable SAM-HQ for better diagram segmentation
USE_SAM_HQ = True  # requires sam_hq_vit_h.pth
```

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, CPU will be slow)
- OpenAI API key with GPT-5 access
- ~4GB VRAM for Florence-2-base, ~8GB for Florence-2-large
- ~2.5GB disk space for SAM checkpoint

## Environment Setup

```bash
export OPENAI_API_KEY='your-key-here'
python main.py slide.png
```

## Advantages over Previous Approach

| Old (Grounding DINO) | New (Florence-2) |
|---------------------|------------------|
| Failed on diagram text | Excels at diagrams |
| Separate OCR needed | Built-in OCR |
| Poor phrase grounding | Native phrase grounding |
| ~40% detection rate | ~80%+ detection rate |
| Multiple dependencies | Single unified model |

