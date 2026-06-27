"""
core/analysis.py
Statistical analysis helpers for processed images.
"""

import cv2
import numpy as np


def sharpness(img_bgr_or_gray: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    if len(img_bgr_or_gray.shape) == 3:
        gray = cv2.cvtColor(img_bgr_or_gray, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr_or_gray
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity (0–255)."""
    return float(gray.mean())


def contrast(gray: np.ndarray) -> float:
    """Std deviation of pixel intensities."""
    return float(gray.std())


def dominant_channel(img_bgr: np.ndarray) -> tuple[str, dict]:
    """
    Return dominant channel label and per-channel means.
    Returns: ('R'|'G'|'B', {'R': float, 'G': float, 'B': float})
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(img_rgb)
    means = {"R": float(r.mean()), "G": float(g.mean()), "B": float(b.mean())}
    return max(means, key=means.get), means


def edge_density(edge_img: np.ndarray) -> dict:
    """
    Analyze a binary edge map (e.g. from Canny).
    Returns dict: total_pixels, edge_pixels, pct, complexity.
    """
    h, w   = edge_img.shape
    total  = h * w
    edge_px = int(np.count_nonzero(edge_img))
    pct    = edge_px / total * 100

    if pct < 5:
        label = "Rendah (objek sederhana)"
    elif pct < 15:
        label = "Sedang (detail cukup)"
    else:
        label = "Tinggi (banyak detail / tekstur)"

    return {
        "total_pixels": total,
        "edge_pixels":  edge_px,
        "pct":          pct,
        "complexity":   label,
    }


def segmentation_stats(binary: np.ndarray) -> dict:
    """Stats for a binary (thresholded) image."""
    h, w   = binary.shape
    total  = h * w
    fg     = int(np.sum(binary == 255))
    bg     = total - fg
    pct_fg = fg / total * 100
    return {
        "total":      total,
        "foreground": fg,
        "background": bg,
        "pct_fg":     pct_fg,
        "dominant":   "Foreground" if pct_fg > 50 else "Background",
    }


def filter_comparison(img_bgr: np.ndarray, filtered_imgs: dict) -> dict:
    """
    Compare sharpness of filtered images vs original.
    Args:
        filtered_imgs: {label: filtered_bgr}
    Returns:
        {original, filters: {label: score}, best: label}
    """
    s_ori  = sharpness(img_bgr)
    scores = {label: sharpness(img) for label, img in filtered_imgs.items()}
    best   = max(scores, key=scores.get)
    return {"original": s_ori, "filters": scores, "best": best}


def histogram_stats(gray_before: np.ndarray, gray_after: np.ndarray) -> dict:
    """Compare contrast before/after histogram equalization."""
    return {
        "mean_before":    float(gray_before.mean()),
        "std_before":     float(gray_before.std()),
        "mean_after":     float(gray_after.mean()),
        "std_after":      float(gray_after.std()),
        "contrast_delta": float(gray_after.std() - gray_before.std()),
    }