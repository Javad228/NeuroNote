"""Configuration settings for the slide analyzer pipeline."""

import torch
from pathlib import Path

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Model configurations -----
# Grounding DINO variants
GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
GROUNDING_DINO_MODEL_DOC = "IDEA-Research/grounding-dino-base"

# Layout-aware detector (for text-heavy diagrams)
LAYOUT_DETECTOR_MODEL = "microsoft/layoutlmv3-base"
LAYOUT_DETECTOR_THRESHOLD = 0.55

# SAM / SAM-HQ configuration
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_HQ_CHECKPOINT = "sam_hq_vit_h.pth"
SAM_HQ_CHECKPOINT_URL = "https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/sam_hq_vit_h.pth"
USE_SAM_HQ = False  # set to True if you have the HQ checkpoint locally

# ----- OCR settings -----
OCR_MIN_CONFIDENCE = 0.55
OCR_MAX_RESULTS = 80
OCR_USE_GPU = torch.cuda.is_available()

# Grounding DINO thresholds
DEFAULT_BOX_THRESHOLD = 0.35
DEFAULT_TEXT_THRESHOLD = 0.25

# Rendering defaults
DEFAULT_FEATHER = 10
DEFAULT_BLUR = 28
DEFAULT_INTENSITY = 0.9
DEFAULT_COLOR_CYCLE = [
    (255, 255, 0),    # Yellow
    (0, 255, 255),    # Cyan
    (255, 0, 255),    # Magenta
    (0, 255, 128),    # Green-cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Blue
]

# GPT-5 Vision settings
GPT5_MODEL = "gpt-5"
GPT5_MAX_TOKENS = 5000
GPT5_REASONING_EFFORT = "low"
GPT5_TEXT_VERBOSITY = "medium"

# Output settings
DEFAULT_OUTPUT_DIR = "out"

# Refinement settings
LINE_REFINEMENT_MIN_POINTS = 50
LINE_REFINEMENT_KMEANS_K = 2
