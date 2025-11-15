"""
GPT-5 Slide Analyzer
====================

Pipeline: GPT-5 Vision → Grounding DINO → SAM
Generates intermediate visualizations at each step.
"""

__version__ = "1.0.0"

from .pipeline import analyze_slide

__all__ = ["analyze_slide"]
