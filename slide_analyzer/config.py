"""Configuration settings for the slide analyzer pipeline."""

import torch
from pathlib import Path

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Model configurations -----
# Florence-2 (Microsoft's vision model for diagrams and OCR)
FLORENCE2_MODEL = "microsoft/Florence-2-large"  # Use base for AMD GPU compatibility
FLORENCE2_OCR_THRESHOLD = 0.5
FLORENCE2_FORCE_CPU = False  # Set to True if GPU has issues

# Legacy models (not used with Florence-2)
# GROUNDING_DINO_MODEL = "IDEA-Research/grounding-dino-tiny"
# LAYOUT_DETECTOR_MODEL = "microsoft/layoutlmv3-base"

# SAM / SAM-HQ configuration
SAM_MODEL_TYPE = "vit_h"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
SAM_CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
SAM_HQ_CHECKPOINT = "sam_hq_vit_h.pth"
SAM_HQ_CHECKPOINT_URL = "https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/sam_hq_vit_h.pth"
USE_SAM_HQ = False  # set to True if you have the HQ checkpoint locally

# Visualization defaults
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
