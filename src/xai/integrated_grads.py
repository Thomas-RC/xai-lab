"""Integrated Gradients (Sundararajan 2017) — przez Captum.

Atrybucja pixel-level po sciezce calkowania od baseline (czarny obraz)
do wejscia. Spelnia aksjomaty Sensitivity i Implementation Invariance.
"""

import time

import numpy as np
import torch
from captum.attr import IntegratedGradients as _IG
from torch import nn

from src.xai.base import XAIMethod, XAIResult, normalize_heatmap


class IntegratedGradients(XAIMethod):
    name = "Integrated Gradients"

    def __init__(self, n_steps: int = 50):
        self.n_steps = n_steps

    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> XAIResult:
        ig = _IG(model)
        baseline = torch.zeros_like(input_tensor)
        t0 = time.perf_counter()
        attribution = ig.attribute(
            input_tensor,
            baselines=baseline,
            target=target_class,
            n_steps=self.n_steps,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # [1, 3, H, W] -> [H, W] przez sumowanie |grad| po kanalach
        heatmap_2d = attribution[0].abs().sum(dim=0)
        heatmap = normalize_heatmap(heatmap_2d)
        return XAIResult(
            method=self.name,
            heatmap=heatmap.astype(np.float32),
            elapsed_ms=elapsed_ms,
            target_class=target_class,
            extra={"n_steps": self.n_steps},
        )
