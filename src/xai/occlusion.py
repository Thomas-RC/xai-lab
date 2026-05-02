"""Occlusion sensitivity (Zeiler & Fergus 2014).

Przesuwamy okno o stalym rozmiarze po obrazie, w kazdej pozycji zastepujemy
fragment szarym pixelem (baseline) i mierzymy spadek prawdopodobienstwa
klasy targetu. Wieksze spadki = bardziej istotny region.

Wlasna implementacja (50 LOC) — dydaktycznie wartosciowa, model-agnostic
(jedyna metoda nie wymagajaca dostepu do gradientow).

Optymalizacja: pozycje grupowane w batche zamiast pojedynczych forwardow.
"""

import time

import numpy as np
import torch
from torch import nn

from src.xai.base import XAIMethod, XAIResult, normalize_heatmap


class Occlusion(XAIMethod):
    name = "Occlusion"

    def __init__(self, patch: int = 32, stride: int = 16, baseline_value: float = 0.0):
        self.patch = patch
        self.stride = stride
        self.baseline_value = baseline_value

    @torch.inference_mode()
    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> XAIResult:
        device = input_tensor.device
        _, _, h, w = input_tensor.shape
        t0 = time.perf_counter()

        baseline_prob = torch.softmax(model(input_tensor), dim=-1)[0, target_class].item()

        ys = list(range(0, h - self.patch + 1, self.stride))
        xs = list(range(0, w - self.patch + 1, self.stride))
        positions = [(y, x) for y in ys for x in xs]

        heatmap = torch.zeros(h, w, device=device)
        counts = torch.zeros(h, w, device=device)

        batch_size = 16
        for i in range(0, len(positions), batch_size):
            batch_positions = positions[i : i + batch_size]
            occluded = input_tensor.repeat(len(batch_positions), 1, 1, 1)
            for j, (y, x) in enumerate(batch_positions):
                occluded[j, :, y : y + self.patch, x : x + self.patch] = self.baseline_value
            probs = torch.softmax(model(occluded), dim=-1)[:, target_class]
            drops = (baseline_prob - probs).clamp(min=0)
            for j, (y, x) in enumerate(batch_positions):
                heatmap[y : y + self.patch, x : x + self.patch] += drops[j]
                counts[y : y + self.patch, x : x + self.patch] += 1

        heatmap = heatmap / counts.clamp(min=1)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return XAIResult(
            method=self.name,
            heatmap=normalize_heatmap(heatmap).astype(np.float32),
            elapsed_ms=elapsed_ms,
            target_class=target_class,
            extra={
                "patch": self.patch,
                "stride": self.stride,
                "n_positions": len(positions),
            },
        )
