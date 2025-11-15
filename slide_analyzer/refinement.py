"""Mask refinement utilities for line detection."""

import numpy as np
import cv2
from PIL import Image

from .config import LINE_REFINEMENT_MIN_POINTS, LINE_REFINEMENT_KMEANS_K


def refine_line_in_mask(image_pil: Image.Image, mask_bool):
    """
    Refine mask to focus on line-like structures using color clustering.
    
    Uses K-means clustering on pixel colors within the mask to separate
    foreground (line) from background.
    """
    img = np.array(image_pil)
    roi = cv2.bitwise_and(img, img, mask=(mask_bool.astype(np.uint8) * 255))

    pts = roi[mask_bool].reshape(-1, 3).astype(np.float32)
    
    if len(pts) < LINE_REFINEMENT_MIN_POINTS:
        return mask_bool

    K = LINE_REFINEMENT_KMEANS_K
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pts, K, None, criteria, 2, cv2.KMEANS_PP_CENTERS
    )
    
    counts = np.bincount(labels.flatten())
    target_cluster = counts.argmin() if len(counts) > 1 else 0

    refined = np.zeros(mask_bool.shape, dtype=np.uint8)
    refined[mask_bool] = (labels.flatten() == target_cluster).astype(np.uint8) * 255
    
    refined = cv2.morphologyEx(
        refined, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    )
    
    return refined > 0
