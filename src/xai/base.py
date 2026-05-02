"""Wspolny interfejs metod XAI.

Kazda metoda dostaje (model, input_tensor, target_class) i zwraca
`XAIResult` z heatmapa [H, W] znormalizowana do [0, 1] oraz metadanymi.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class XAIResult:
    method: str
    heatmap: np.ndarray
    """[H, W] float32 in [0, 1] — znormalizowana mapa istotnosci."""
    elapsed_ms: float
    target_class: int
    extra: dict | None = None


class XAIMethod(ABC):
    name: str = "base"

    @abstractmethod
    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> XAIResult: ...


def normalize_heatmap(arr: np.ndarray | torch.Tensor) -> np.ndarray:
    """Klamruje do [0, max] i skaluje do [0, 1]. Zwraca float32 [H, W]."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = arr.astype(np.float32)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    peak = float(arr.max())
    if peak > 1e-8:
        arr = arr / peak
    return arr
