"""IoU (Intersection over Union) dla heatmap parami.

Heatmapy nie sa maskami binarnymi — przed liczeniem IoU thresholdujemy je:
biorac top-K% pikseli wedlug istotnosci. Domyslnie 20% (typowe w XAI papers,
np. Adebayo 2018).

Symetryczny: IoU(A, B) = IoU(B, A). Zakres [0, 1] gdzie 1 = identyczne maski.
"""

import numpy as np
from PIL import Image


def _to_same_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape == b.shape:
        return a, b
    h, w = a.shape
    b_resized = Image.fromarray((b * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    return a, np.asarray(b_resized, dtype=np.float32) / 255.0


def top_k_mask(heatmap: np.ndarray, top_pct: float = 0.20) -> np.ndarray:
    """Zwraca maske binarna: True dla top top_pct% pikseli wedlug intensywnosci."""
    flat = heatmap.flatten()
    k = max(1, int(len(flat) * top_pct))
    threshold = np.partition(flat, -k)[-k]
    return heatmap >= threshold


def iou(a: np.ndarray, b: np.ndarray, top_pct: float = 0.20) -> float:
    a, b = _to_same_shape(a, b)
    mask_a = top_k_mask(a, top_pct)
    mask_b = top_k_mask(b, top_pct)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def pairwise_iou(heatmaps: list[np.ndarray], top_pct: float = 0.20) -> np.ndarray:
    """Macierz [N, N] symetryczna z 1 na diagonali."""
    n = len(heatmaps)
    matrix = np.eye(n, dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            value = iou(heatmaps[i], heatmaps[j], top_pct)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix
