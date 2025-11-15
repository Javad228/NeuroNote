"""Command-line entry point for the slide analyzer."""

import sys
import argparse
from pathlib import Path

# Handle both module and direct execution
if __name__ == "__main__" and __package__ is None:
    # Running directly, add parent directory to path
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from slide_analyzer.pipeline import analyze_slide
else:
    # Running as module
    from .pipeline import analyze_slide


def main():
    """CLI interface for analyzing slides."""
    parser = argparse.ArgumentParser(
        description="Analyze educational slides using GPT-5 Vision, Grounding DINO, and SAM"
    )
    
    parser.add_argument("image", help="Path to slide image")
    parser.add_argument("--output", "-o", default="out", help="Output directory (default: out)")
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        sys.exit(1)
    
    try:
        result = analyze_slide(
            args.image,
            openai_api_key=None,  # Uses OPENAI_API_KEY env var
            output_dir=args.output
        )
        
        print(f"\n✅ Success! Results saved to {args.output}/")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
