"""Rendering utilities for glow effects."""

import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

from .config import DEFAULT_FEATHER, DEFAULT_BLUR, DEFAULT_INTENSITY, DEFAULT_COLOR_CYCLE
from .image_utils import rect_mask
from .sam_segmentation import box_to_best_mask
from .refinement import refine_line_in_mask


def glow_composite(base_rgb: Image.Image, mask_L: Image.Image, 
                  blur=DEFAULT_BLUR, intensity=DEFAULT_INTENSITY, 
                  color=(255, 255, 0)):
    """Apply a glow effect over the base image using a mask."""
    blurred = mask_L.filter(ImageFilter.GaussianBlur(blur))
    
    a = np.array(blurred, dtype=np.float32) * float(intensity)
    a = np.clip(a, 0, 255).astype(np.uint8)
    
    arr = np.zeros((base_rgb.height, base_rgb.width, 4), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    arr[..., 3] = a
    
    overlay = Image.fromarray(arr, "RGBA")
    out = Image.alpha_composite(base_rgb.convert("RGBA"), overlay)
    return out.convert("RGB")


def render_glows(image_pil: Image.Image, refined_items, sam_pred,
                feather=DEFAULT_FEATHER, blur=DEFAULT_BLUR, 
                intensity=DEFAULT_INTENSITY,
                color_cycle=None, prefer_line=True):
    """Render glow highlights for all refined items."""
    if color_cycle is None:
        color_cycle = DEFAULT_COLOR_CYCLE
    
    W, H = image_pil.size
    out = image_pil.copy()
    frames = []

    print(f"\nRendering glow highlights...")
    
    for idx, (s, xyxy, score) in enumerate(refined_items):
        sam_mask_bool = box_to_best_mask(image_pil, xyxy, sam_pred)

        if sam_mask_bool.sum() < 30:
            mask_L = rect_mask((W, H), xyxy, feather=feather)
        else:
            if prefer_line:
                line_mask_bool = refine_line_in_mask(image_pil, sam_mask_bool)
                if line_mask_bool.sum() > 30:
                    mask_bool_to_render = line_mask_bool
                else:
                    mask_bool_to_render = sam_mask_bool
            else:
                mask_bool_to_render = sam_mask_bool

            mask_L = Image.fromarray(
                (mask_bool_to_render.astype(np.uint8) * 255), 
                mode="L"
            )
            mask_L = mask_L.filter(ImageFilter.GaussianBlur(radius=2))

        color = color_cycle[idx % len(color_cycle)]
        out = glow_composite(out, mask_L, blur=blur, intensity=intensity, color=color)

        dbg = out.copy()
        draw = ImageDraw.Draw(dbg)
        x1, y1, x2, y2 = map(int, xyxy)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        frames.append(dbg)

    print(f"✓ Rendered {len(refined_items)} glow highlights")
    return out, frames


def render_each_sentence_image(image_pil: Image.Image, refined_items, sam_pred,
                               save_dir="out/sentences", 
                               feather=DEFAULT_FEATHER,
                               blur=DEFAULT_BLUR, 
                               intensity=DEFAULT_INTENSITY):
    """Create a separate glow-highlighted image for each sentence."""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    color_cycle = DEFAULT_COLOR_CYCLE

    print(f"\nCreating individual sentence images...")
    results = []
    
    for idx, (s, xyxy, score) in enumerate(refined_items):
        mask_bool = box_to_best_mask(image_pil, xyxy, sam_pred)

        if mask_bool.sum() < 30:
            mask_L = rect_mask(image_pil.size, xyxy, feather=feather)
        else:
            mask_L = Image.fromarray(
                (mask_bool.astype(np.uint8) * 255), 
                mode="L"
            )
            mask_L = mask_L.filter(ImageFilter.GaussianBlur(radius=2))

        color = color_cycle[idx % len(color_cycle)]
        highlighted = glow_composite(
            image_pil, mask_L, blur=blur, intensity=intensity, color=color
        )

        filename = Path(save_dir) / f"sentence_{idx+1:02d}.png"
        highlighted.save(filename)
        results.append((s["text"], str(filename)))

    print(f"✓ Created {len(results)} individual images")
    return results
