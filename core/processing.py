import os
import cv2
import numpy as np
from pathlib import Path

_CORE_DIR    = Path(__file__).parent
_DNN_PROTO   = str(_CORE_DIR / "deploy.prototxt")
_DNN_MODEL   = str(_CORE_DIR / "res10_face.caffemodel")

_dnn_net = None

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def to_binary(img_bgr: np.ndarray, threshold: int = 127) -> np.ndarray:
    gray = to_grayscale(img_bgr)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def equalize_histogram(img_bgr: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img_bgr)
    return cv2.equalizeHist(gray)

def contrast_stretching(img_bgr: np.ndarray) -> np.ndarray:
    result = np.zeros_like(img_bgr)
    for i in range(3):
        ch = img_bgr[:, :, i].astype(np.float32)
        mn, mx = ch.min(), ch.max()
        if mx > mn:
            result[:, :, i] = ((ch - mn) / (mx - mn) * 255).astype(np.uint8)
        else:
            result[:, :, i] = ch.astype(np.uint8)
    return result


def brightness_adjustment(img_bgr: np.ndarray, beta: int = 50) -> np.ndarray:
    return cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=beta)


def sharpening(img_bgr: np.ndarray, strength: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 3)
    return cv2.addWeighted(img_bgr, 1 + strength, blurred, -strength, 0)

def filter_mean(img_bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.blur(img_bgr, (ksize, ksize))


def filter_median(img_bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    return cv2.medianBlur(img_bgr, ksize)


def filter_gaussian(img_bgr: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img_bgr, (k, k), sigma)

def edge_canny(img_bgr: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    gray = to_grayscale(img_bgr)
    return cv2.Canny(gray, low, high)


def edge_prewitt(img_bgr: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img_bgr)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    return np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)


def edge_sobel(img_bgr: np.ndarray) -> np.ndarray:
    gray = to_grayscale(img_bgr)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)

def segment_otsu(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    gray = to_grayscale(img_bgr)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(ret)


def segment_adaptive(img_bgr: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    gray = to_grayscale(img_bgr)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )

def segment_kmeans(img_bgr: np.ndarray, k: int = 3) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    data = img_bgr.reshape((-1, 3)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, attempts=5,
        flags=cv2.KMEANS_RANDOM_CENTERS
    )
    centers = centers.astype(np.uint8)
    result  = centers[labels.flatten()].reshape((h, w, 3))
    return result


def segment_watershed(img_bgr: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel  = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers     = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr, markers)
    result  = img_bgr.copy()
    result[markers == -1] = [0, 0, 255]   
    return result
