"""Wizualizacja heatmap: nakladka na obraz i galeria 2x3."""

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from PIL import Image

from src.xai.base import XAIResult


def overlay_heatmap(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """rgb: [H, W, 3] in [0,1]; heatmap: [H', W'] in [0,1].
    Heatmapa jest upsamplowana do rozmiaru rgb przed nalozeniem.
    Zwraca [H, W, 3] uint8.
    """
    h, w, _ = rgb.shape
    if heatmap.shape != (h, w):
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
            (w, h), Image.BILINEAR
        )
        heatmap_resized = np.asarray(heatmap_img, dtype=np.float32) / 255.0
    else:
        heatmap_resized = heatmap

    cmap = cm.get_cmap(colormap)
    colored = cmap(heatmap_resized)[..., :3]
    blended = (1 - alpha) * rgb + alpha * colored
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def gallery_figure(
    rgb: np.ndarray,
    results: list[XAIResult],
    title: str | None = None,
    cols: int = 3,
) -> Figure:
    """Galeria heatmap obok siebie (2x3 dla 6 metod) + obraz oryginalny + tytul."""
    n = len(results)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows + 1, cols, figsize=(cols * 3.5, (rows + 1) * 3.5))

    if axes.ndim == 1:
        axes = axes.reshape(1, -1)

    # Pierwszy rzad: oryginalny obraz wycentrowany
    for ax in axes[0]:
        ax.axis("off")
    axes[0, cols // 2].imshow(rgb)
    axes[0, cols // 2].set_title("Oryginał", fontsize=11)

    # Reszta: heatmapy
    for i, result in enumerate(results):
        r = 1 + i // cols
        c = i % cols
        overlay = overlay_heatmap(rgb, result.heatmap)
        axes[r, c].imshow(overlay)
        axes[r, c].set_title(f"{result.method} ({result.elapsed_ms:.0f} ms)", fontsize=10)
        axes[r, c].axis("off")

    # Wygas niewykorzystane subploty
    for i in range(n, rows * cols):
        r = 1 + i // cols
        c = i % cols
        axes[r, c].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
