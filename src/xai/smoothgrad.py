"""SmoothGrad (Smilkov 2017) — usrednianie gradientow z N zaszumionych kopii wejscia.

Implementacja: Captum NoiseTunnel z bazowa metoda Saliency (zwykly gradient).
NoiseTunnel z Saliency przybliza klasyczny SmoothGrad. Mozna tez owinac IG
(SmoothGrad-IG / SGSQ) — robimy klasyk dla czystej referencji.
"""

import time

import numpy as np
import torch
from captum.attr import NoiseTunnel, Saliency
from torch import nn

from src.xai.base import XAIMethod, XAIResult, normalize_heatmap


class SmoothGrad(XAIMethod):
    name = "SmoothGrad"

    def __init__(self, n_samples: int = 50, sigma: float = 0.15):
        self.n_samples = n_samples
        self.sigma = sigma

    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> XAIResult:
        nt = NoiseTunnel(Saliency(model))
        t0 = time.perf_counter()
        attribution = nt.attribute(
            input_tensor,
            target=target_class,
            nt_samples=self.n_samples,
            nt_samples_batch_size=8,
            nt_type="smoothgrad",
            stdevs=self.sigma,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        heatmap_2d = attribution[0].abs().sum(dim=0)
        heatmap = normalize_heatmap(heatmap_2d)
        return XAIResult(
            method=self.name,
            heatmap=heatmap.astype(np.float32),
            elapsed_ms=elapsed_ms,
            target_class=target_class,
            extra={"n_samples": self.n_samples, "sigma": self.sigma},
        )
