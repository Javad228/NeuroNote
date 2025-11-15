"""Visualization utilities for bounding boxes."""

from PIL import Image, ImageDraw, ImageFont
from .config import DEFAULT_COLOR_CYCLE


def draw_bounding_boxes(image_pil: Image.Image, boxes, labels=None, 
                       color_cycle=None, line_width=3):
    """
    Draw bounding boxes on an image.
    
    Args:
        image_pil: PIL Image
        boxes: List of [x1, y1, x2, y2] bounding boxes
        labels: Optional list of text labels for each box
        color_cycle: List of RGB color tuples (uses default if None)
        line_width: Width of bounding box lines
        
    Returns:
        PIL.Image.Image: Image with boxes drawn
    """
    if color_cycle is None:
        color_cycle = DEFAULT_COLOR_CYCLE
    
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    
    # Try to load a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except Exception:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = color_cycle[i % len(color_cycle)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Draw label if provided
        if labels and i < len(labels):
            label = str(labels[i])
            # Draw background for text
            bbox = draw.textbbox((x1 + 5, y1 + 5), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1 + 5, y1 + 5), label, fill=(0, 0, 0), font=font)
    
    return img


def get_mask_bounding_box(mask_bool):
    """
    Get bounding box from a boolean mask.
    
    Args:
        mask_bool: Boolean mask array (H, W)
        
    Returns:
        list: [x1, y1, x2, y2] bounding box, or None if mask is empty
    """
    import numpy as np
    
    if mask_bool.sum() == 0:
        return None
    
    rows = np.any(mask_bool, axis=1)
    cols = np.any(mask_bool, axis=0)
    
    if not rows.any() or not cols.any():
        return None
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return [int(x1), int(y1), int(x2), int(y2)]

