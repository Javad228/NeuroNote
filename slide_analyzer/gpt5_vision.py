"""
GPT-5 Vision integration for slide analysis.
"""

import json
import re
import os
from openai import OpenAI

from .config import GPT5_MODEL, GPT5_MAX_TOKENS, GPT5_REASONING_EFFORT, GPT5_TEXT_VERBOSITY


def setup_openai_client(api_key=None):
    """
    Setup OpenAI client.
    
    Args:
        api_key: OpenAI API key (optional if set in OPENAI_API_KEY environment variable)
        
    Returns:
        OpenAI: Configured client
    """
    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError(
            "Please provide OpenAI API key via parameter or "
            "set OPENAI_API_KEY environment variable"
        )
    
    client = OpenAI(api_key=api_key)
    print("✓ OpenAI client ready")
    return client


def _parse_gpt5_sentences_json(content: str):
    """
    Parse JSON response from GPT-5, handling various formats.
    
    Args:
        content: Raw response text
        
    Returns:
        dict: Parsed JSON data
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
    # Try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Try fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try to salvage the sentences array
    m = re.search(r'"sentences"\s*:\s*\[(.*)', content, re.DOTALL)
    if not m:
        raise ValueError("Could not find 'sentences' in GPT-5 output.")
    
    body = m.group(1)
    trimmed = body.rsplit("}", 1)[0] + "}"
    trimmed = re.sub(r"}\s*{", "},{", trimmed)
    salvage = "{\n  \"sentences\": [" + trimmed + "]\n}"
    
    try:
        return json.loads(salvage)
    except Exception as e:
        raise ValueError(
            f"Failed to salvage GPT-5 JSON: {e}\n"
            f"First 500 chars:\n{content[:500]}"
        )


def _clamp_coordinate(value, default=0.5):
    """
    Clamp coordinate value to [0.0, 1.0] range.
    
    Args:
        value: Input value
        default: Default value if parsing fails
        
    Returns:
        float: Clamped value
    """
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def generate_explanation_with_gpt5(image_base64, client, ocr_snippets=None):
    """
    Use GPT-5 Vision to analyze a slide and produce explanatory sentences
    with bounding boxes.
    
    Args:
        image_base64: Base64 encoded PNG image
        client: OpenAI client instance
        
    Returns:
        tuple: (sentences, raw_content)
            - sentences: List of dicts with keys: text, x, y, width, height
            - raw_content: Raw text response from GPT-5
              
    Raises:
        Exception: If GPT-5 analysis fails
    """
    recognized_text_block = ""
    if ocr_snippets:
        trimmed = [txt for txt in ocr_snippets if txt.strip()]
        trimmed = trimmed[:30]
        if trimmed:
            bullet_list = "\n".join(f"- {t}" for t in trimmed)
            recognized_text_block = (
                "\nRecognized text snippets on the slide:\n"
                f"{bullet_list}\n"
            )

    prompt_text = (
        "You are an expert educator analyzing a lecture slide. "
        "Your goal is to EXPLAIN the concepts illustrated, not just describe what you see visually.\n\n"
        "Write 5-8 sentences that teach the key concepts shown on this slide. Each sentence should:\n"
        "- Explain a concept, principle, or relationship illustrated in the slide\n"
        "- Be pedagogical—teach as if to a student learning this material\n"
        "- Reference WHERE the concept is shown (so students know what to look at)\n\n"
        "Use the real text anchors extracted via OCR to avoid hallucinations.\n"
        f"{recognized_text_block}\n"
        "For each explanatory sentence, provide:\n"
        '1. \"text\": The full explanatory sentence\n'
        '2. \"detection_phrases\": A JSON array with 1-3 entries. EACH entry must:\n'
        "   - Be 2-5 words\n"
        "   - Contain the object type + color + text label (if present)\n"
        "   - Be concrete, uniquely identifiable, and omit commas\n"
        "   - Examples (GOOD): \"blue box 'Mesos master'\", \"green circles 'ZooKeeper quorum'\", \"down arrows from master\"\n"
        "   - Examples (BAD): \"the title\", \"blue box\", \"arrow\", \"text label\", \"main component\"\n"
        "3. Bounding box coordinates:\n"
        "   - x: horizontal center (0.0 = left edge, 1.0 = right edge)\n"
        "   - y: vertical center (0.0 = top edge, 1.0 = bottom edge)\n"
        "   - width: box width (0.0–1.0)\n"
        "   - height: box height (0.0–1.0)\n\n"
        "Return JSON only:\n"
        "{{\n"
        '  "sentences": [\n'
        "    {{\n"
        '      "text": "The title at the top, \'Slide Topic\', introduces the main concept.",\n'
        '      "detection_phrases": ["title text \'Slide Topic\'"],\n'
        '      "x": 0.5,\n'
        '      "y": 0.1,\n'
        '      "width": 0.8,\n'
        '      "height": 0.15\n'
        "    }},\n"
        "    {{\n"
        '      "text": "The central \'Core Component\' box is the primary element in this diagram.",\n'
        '      "detection_phrases": ["blue box \'Core Component\'"],\n'
        '      "x": 0.5,\n'
        '      "y": 0.4,\n'
        '      "width": 0.3,\n'
        '      "height": 0.2\n'
        "    }},\n"
        "    {{\n"
        '      "text": "The arrows pointing from the main box show the process flow.",\n'
        '      "detection_phrases": ["down arrows from master"],\n'
        '      "x": 0.5,\n'
        '      "y": 0.6,\n'
        '      "width": 0.4,\n'
        '      "height": 0.2\n'
        "    }}\n"
        "  ]\n"
        "}}\n"
    )
    try:
        print("🔍 Starting GPT-5 Vision analysis...")
        
        response = client.responses.create(
            model=GPT5_MODEL,
            input=[{
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    },
                    {
                        "type": "input_text",
                        "text": prompt_text
                    }
                ]
            }],
            text={"verbosity": GPT5_TEXT_VERBOSITY},
            reasoning={"effort": GPT5_REASONING_EFFORT},
            max_output_tokens=GPT5_MAX_TOKENS
        )

        # Extract plain text from the multimodal response
        content = None
        if hasattr(response, "output") and response.output:
            for item in response.output:
                if getattr(item, "type", "") != "reasoning":
                    if hasattr(item, "text") and item.text:
                        content = item.text
                        break
                    if hasattr(item, "content") and isinstance(item.content, list) and item.content:
                        first = item.content[0]
                        if hasattr(first, "text"):
                            content = first.text
                            break
        
        if not content and hasattr(response, "output_text"):
            content = response.output_text
        
        if not content or not content.strip():
            raise ValueError("GPT-5 returned empty response.")

        # Parse and validate
        data = _parse_gpt5_sentences_json(content)
        sentences = []
        
        for it in data.get("sentences", []):
            if "text" not in it:
                continue
            
            # Extract detection phrases (preferred) or keywords
            raw_phrases = it.get("detection_phrases")
            if isinstance(raw_phrases, list):
                detection_phrases = [str(p).strip() for p in raw_phrases if str(p).strip()]
            elif isinstance(raw_phrases, str):
                detection_phrases = [p.strip() for p in raw_phrases.split(",") if p.strip()]
            else:
                detection_phrases = []

            # Fallback: use provided keywords string
            if not detection_phrases:
                keywords_fallback = it.get("keywords", "")
                if isinstance(keywords_fallback, str):
                    detection_phrases = [p.strip() for p in keywords_fallback.split(",") if p.strip()]
                elif isinstance(keywords_fallback, list):
                    detection_phrases = [str(p).strip() for p in keywords_fallback if str(p).strip()]

            # Final fallback: first few words of the text
            if not detection_phrases:
                text_words = str(it["text"]).split()[:4]
                fallback_phrase = " ".join(text_words).strip()
                if fallback_phrase:
                    detection_phrases = [fallback_phrase]

            sentences.append({
                "text": str(it["text"]).strip(),
                "detection_phrases": detection_phrases,
                "keywords": ", ".join(detection_phrases),
                "x": _clamp_coordinate(it.get("x", 0.5), 0.5),
                "y": _clamp_coordinate(it.get("y", 0.5), 0.5),
                "width": _clamp_coordinate(it.get("width", 0.3), 0.3),
                "height": _clamp_coordinate(it.get("height", 0.15), 0.15),
            })
        
        print(f"✓ GPT-5 generated {len(sentences)} explanatory sentences with keywords")
        return sentences, content

    except Exception as e:
        raise Exception(f"GPT-5 Vision analysis failed: {e}")

