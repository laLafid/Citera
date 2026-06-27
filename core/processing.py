"""
core/processing.py
All image processing operations, returning numpy arrays.
Each function accepts a BGR numpy array (as returned by cv2.imread).

Face detection methods:
  - 'haar'  : Multi-cascade Haar (bundled, no model file needed)
  - 'dnn'   : OpenCV DNN / SSD ResNet10 (needs core/deploy.prototxt + core/res10_face.caffemodel)
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Path to DNN model files (relative to this file's directory)
_CORE_DIR    = Path(__file__).parent
_DNN_PROTO   = str(_CORE_DIR / "deploy.prototxt")
_DNN_MODEL   = str(_CORE_DIR / "res10_face.caffemodel")

# Cached DNN net (loaded once)
_dnn_net = None


# ── Color Conversion ─────────────────────────────────────────────

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """BGR → RGB."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def to_grayscale(img_bgr: np.ndarray) -> np.ndarray:
    """BGR → Grayscale (single-channel)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def to_binary(img_bgr: np.ndarray, threshold: int = 127) -> np.ndarray:
    """BGR → Binary (fixed threshold)."""
    gray = to_grayscale(img_bgr)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


# ── Perbaikan Kualitas Tambahan ──────────────────────────────────

def equalize_histogram(img_bgr: np.ndarray) -> np.ndarray:
    """Grayscale histogram equalization. Returns grayscale uint8."""
    gray = to_grayscale(img_bgr)
    return cv2.equalizeHist(gray)

def contrast_stretching(img_bgr: np.ndarray) -> np.ndarray:
    """
    Contrast stretching (normalisasi min-max per channel).
    Meregangkan rentang intensitas ke penuh 0-255.
    """
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
    """
    Brightness adjustment via addWeighted.
    beta > 0 = lebih terang, beta < 0 = lebih gelap. Range: -255..255.
    """
    return cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=beta)


def sharpening(img_bgr: np.ndarray, strength: float = 1.5) -> np.ndarray:
    """
    Unsharp masking sharpening.
    strength: seberapa kuat efek tajam (1.0 = ringan, 2.0+ = kuat).
    """
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), 3)
    return cv2.addWeighted(img_bgr, 1 + strength, blurred, -strength, 0)


# ── Filtering ────────────────────────────────────────────────────

def filter_mean(img_bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Mean (box) blur. Returns BGR."""
    return cv2.blur(img_bgr, (ksize, ksize))


def filter_median(img_bgr: np.ndarray, ksize: int = 5) -> np.ndarray:
    """Median blur. Returns BGR."""
    return cv2.medianBlur(img_bgr, ksize)


def filter_gaussian(img_bgr: np.ndarray, ksize: int = 5, sigma: float = 0) -> np.ndarray:
    """Gaussian blur. Returns BGR."""
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img_bgr, (k, k), sigma)


# ── Edge Detection ───────────────────────────────────────────────

def edge_canny(img_bgr: np.ndarray, low: int = 100, high: int = 200) -> np.ndarray:
    """Canny edge detection. Returns single-channel uint8."""
    gray = to_grayscale(img_bgr)
    return cv2.Canny(gray, low, high)


def edge_prewitt(img_bgr: np.ndarray) -> np.ndarray:
    """Prewitt edge detection. Returns single-channel uint8."""
    gray = to_grayscale(img_bgr)
    kx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
    gx = cv2.filter2D(gray, cv2.CV_32F, kx)
    gy = cv2.filter2D(gray, cv2.CV_32F, ky)
    return np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)


def edge_sobel(img_bgr: np.ndarray) -> np.ndarray:
    """Sobel edge detection. Returns single-channel uint8."""
    gray = to_grayscale(img_bgr)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.clip(np.sqrt(gx**2 + gy**2), 0, 255).astype(np.uint8)


# ── Segmentation ─────────────────────────────────────────────────

def segment_otsu(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Otsu thresholding. Returns (binary_image, threshold_value)."""
    gray = to_grayscale(img_bgr)
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary, float(ret)


def segment_adaptive(img_bgr: np.ndarray, block_size: int = 11, C: int = 2) -> np.ndarray:
    """Adaptive thresholding — better for uneven lighting."""
    gray = to_grayscale(img_bgr)
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C
    )

def segment_kmeans(img_bgr: np.ndarray, k: int = 3) -> np.ndarray:
    """
    K-Means color segmentation.
    k: jumlah cluster warna (default 3).
    Returns: BGR image dengan tiap pixel diganti warna centroid cluster-nya.
    """
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
    """
    Watershed segmentation.
    Returns: BGR image dengan batas segmen ditandai warna merah.
    """
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Noise removal
    kernel  = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background & foreground
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    dist    = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)

    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    markers     = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(img_bgr, markers)
    result  = img_bgr.copy()
    result[markers == -1] = [0, 0, 255]   # batas = merah
    return result


# ── Face Detection ───────────────────────────────────────────────

def _load_dnn_net():
    """Lazily load the DNN face detection network."""
    global _dnn_net
    if _dnn_net is None:
        if not os.path.exists(_DNN_PROTO) or not os.path.exists(_DNN_MODEL):
            raise FileNotFoundError(
                f"Model DNN tidak ditemukan.\n"
                f"Pastikan file berikut ada di folder core/:\n"
                f"  - deploy.prototxt\n"
                f"  - res10_face.caffemodel\n"
            )
        _dnn_net = cv2.dnn.readNetFromCaffe(_DNN_PROTO, _DNN_MODEL)
    return _dnn_net


def detect_faces_haar(
    img_bgr: np.ndarray,
    min_neighbors: int = 6,
    min_size: tuple[int, int] = (40, 40),
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Multi-cascade Haar face detection.
    Uses frontal (default + alt + alt2) and profile cascades,
    then merges overlapping detections via groupRectangles.

    Returns:
        result_bgr: Image with rectangles drawn (green = frontal, blue = profile).
        faces:      List of (x, y, w, h).
    """
    gray = to_grayscale(img_bgr)
    # Equalize for better detection in varying light
    gray_eq = cv2.equalizeHist(gray)

    cascade_configs = [
        (cv2.data.haarcascades + "haarcascade_frontalface_default.xml", (0, 220, 0),   1.05, min_neighbors),
        (cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml",    (0, 200, 50),  1.05, min_neighbors - 1),
        (cv2.data.haarcascades + "haarcascade_profileface.xml",         (255, 180, 0), 1.05, min_neighbors),
    ]

    all_rects: list[tuple[int, int, int, int]] = []
    for path, _, scale, neighbors in cascade_configs:
        cc = cv2.CascadeClassifier(path)
        det = cc.detectMultiScale(gray_eq, scaleFactor=scale,
                                  minNeighbors=neighbors, minSize=min_size)
        if len(det) > 0:
            all_rects.extend([tuple(d) for d in det])

    # Merge overlapping boxes
    faces = _merge_rects(all_rects, overlap_thresh=0.3)

    result = img_bgr.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 220, 0), 2)
        cv2.putText(result, "face", (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)

    return result, faces


def detect_faces_dnn(
    img_bgr: np.ndarray,
    confidence_threshold: float = 0.5,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    OpenCV DNN face detection (SSD ResNet-10, much more accurate than Haar).
    Requires core/deploy.prototxt and core/res10_face.caffemodel.

    Returns:
        result_bgr: Image with bounding boxes drawn.
        faces:      List of (x, y, w, h).
    """
    net = _load_dnn_net()
    h, w = img_bgr.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(img_bgr, (300, 300)), 1.0, (300, 300), (104, 117, 123)
    )
    net.setInput(blob)
    detections = net.forward()  # shape: (1, 1, 200, 7)

    faces: list[tuple[int, int, int, int]] = []
    result = img_bgr.copy()

    for i in range(detections.shape[2]):
        conf = float(detections[0, 0, i, 2])
        if conf < confidence_threshold:
            continue

        x1 = int(detections[0, 0, i, 3] * w)
        y1 = int(detections[0, 0, i, 4] * h)
        x2 = int(detections[0, 0, i, 5] * w)
        y2 = int(detections[0, 0, i, 6] * h)

        # Clamp to image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        fw, fh  = x2 - x1, y2 - y1
        if fw <= 0 or fh <= 0:
            continue

        faces.append((x1, y1, fw, fh))
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 180, 255), 2)
        label = f"{conf:.0%}"
        cv2.putText(result, label, (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1)

    return result, faces


def detect_faces(
    img_bgr: np.ndarray,
    method: str = "dnn",
    **kwargs,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Unified face detection dispatcher.

    Args:
        img_bgr: Input image in BGR.
        method:  'dnn' (default, more accurate) or 'haar' (no model file needed).
        **kwargs: Passed to the underlying detector.

    Returns:
        (result_bgr, faces_list)
    """
    if method == "dnn":
        try:
            return detect_faces_dnn(img_bgr, **kwargs)
        except FileNotFoundError:
            # Graceful fallback
            return detect_faces_haar(img_bgr, **kwargs)
    return detect_faces_haar(img_bgr, **kwargs)


def process_frame_realtime(
    frame_bgr: np.ndarray,
    method: str = "dnn",
    confidence_threshold: float = 0.5,
    min_neighbors: int = 6,
) -> tuple[np.ndarray, int]:
    """
    Optimized face detection for real-time video frames.
    Downscales internally for speed, draws on original resolution.

    Returns:
        (annotated_frame_bgr, face_count)
    """
    h, w = frame_bgr.shape[:2]

    if method == "dnn":
        try:
            net = _load_dnn_net()
            # DNN works at 300x300 anyway, no extra resize needed
            result, faces = detect_faces_dnn(frame_bgr, confidence_threshold)
            return result, len(faces)
        except FileNotFoundError:
            method = "haar"

    # Haar: process at half-res for speed, scale boxes back up
    scale = 0.5
    small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    _, faces_small = detect_faces_haar(small, min_neighbors=min_neighbors)

    result = frame_bgr.copy()
    for (x, y, fw, fh) in faces_small:
        x2 = int(x / scale); y2 = int(y / scale)
        w2 = int(fw / scale); h2 = int(fh / scale)
        cv2.rectangle(result, (x2, y2), (x2 + w2, y2 + h2), (0, 220, 0), 2)

    return result, len(faces_small)


# ── Internal helpers ─────────────────────────────────────────────

def _iou(a: tuple, b: tuple) -> float:
    """Intersection-over-Union for two (x,y,w,h) rects."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _merge_rects(
    rects: list[tuple], overlap_thresh: float = 0.3
) -> list[tuple[int, int, int, int]]:
    """Simple greedy NMS-style merge for overlapping rectangles."""
    if not rects:
        return []

    # Sort by area descending
    rects = sorted(rects, key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    used = [False] * len(rects)

    for i, r in enumerate(rects):
        if used[i]:
            continue
        kept.append(r)
        for j in range(i + 1, len(rects)):
            if not used[j] and _iou(r, rects[j]) > overlap_thresh:
                used[j] = True

    return [tuple(int(v) for v in r) for r in kept]