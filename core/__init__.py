"""core/__init__.py"""

from .loader import load_images, load_faces_or_dataset, ImageNavigator
from .processing import (
    to_rgb, to_grayscale, to_binary,
    equalize_histogram,
    filter_mean, filter_median, filter_gaussian,
    edge_canny, edge_prewitt, edge_sobel,
    segment_otsu, segment_adaptive,
    contrast_stretching, brightness_adjustment, sharpening,
    segment_kmeans, segment_watershed,
)
from .analysis import (
    sharpness, brightness, contrast,
    dominant_channel, edge_density,
    segmentation_stats, filter_comparison, histogram_stats,
)

__all__ = [
    "load_images", "load_faces_or_dataset", "ImageNavigator",
    "to_rgb", "to_grayscale", "to_binary",
    "equalize_histogram",
    "filter_mean", "filter_median", "filter_gaussian",
    "edge_canny", "edge_prewitt", "edge_sobel",
    "segment_otsu", "segment_adaptive",
    "contrast_stretching", "brightness_adjustment", "sharpening",
    "segment_kmeans", "segment_watershed",
    "sharpness", "brightness", "contrast",
    "dominant_channel", "edge_density",
    "segmentation_stats", "filter_comparison", "histogram_stats",
]