"""Image loading, processing, and utility functions."""

import io
import base64
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw


def load_image_from_path_or_b64(image_input):
    """Load image from file path, bytes, or base64 string."""
    if isinstance(image_input, str) and Path(image_input).exists():
        return Image.open(image_input).convert("RGB")
    
    if isinstance(image_input, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_input)).convert("RGB")
    
    if isinstance(image_input, str):
        if image_input.startswith("data:image"):
            _, b64 = image_input.split(",", 1)
        else:
            b64 = image_input
        pad = len(b64) % 4
        if pad:
            b64 += "=" * (4 - pad)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    
    raise ValueError("Unsupported image input")


def pil_to_base64_png(im: Image.Image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_xywh_to_xyxy(x, y, w, h, W, H):
    """Convert normalized center coordinates to pixel coordinates."""
    cx, cy = x * W, y * H
    ww, hh = w * W, h * H
    x1, y1 = int(max(0, cx - ww/2)), int(max(0, cy - hh/2))
    x2, y2 = int(min(W-1, cx + ww/2)), int(min(H-1, cy + hh/2))
    return [x1, y1, x2, y2]


def rect_mask(size_wh, xyxy, feather=0):
    """Create a rectangular mask with optional feathering."""
    W, H = size_wh
    x1, y1, x2, y2 = map(int, xyxy)
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    d.rectangle([x1, y1, x2, y2], fill=255)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather))
    return m
