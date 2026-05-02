"""Grad-CAM (Selvaraju 2017) i Grad-CAM++ (Chattopadhyay 2018).

Implementacja przez `pytorch-grad-cam` (jacobgil) — biblioteka obsluguje
zarowno CNN (target_layer = ostatnia conv) jak i ViT (target_layer = ln_1
ostatniego encoder block + reshape_transform tokenow patchy do 14x14).
"""

import time
from typing import Callable

import numpy as np
import torch
from pytorch_grad_cam import GradCAM as _GradCAM
from pytorch_grad_cam import GradCAMPlusPlus as _GradCAMpp
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn

from src.xai.base import XAIMethod, XAIResult, normalize_heatmap


class _GradCAMBase(XAIMethod):
    cam_class: type
    name: str

    def __init__(
        self,
        target_layer: nn.Module,
        reshape_transform: Callable | None = None,
    ):
        self.target_layer = target_layer
        self.reshape_transform = reshape_transform

    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> XAIResult:
        targets = [ClassifierOutputTarget(target_class)]
        t0 = time.perf_counter()
        cam = self.cam_class(
            model=model,
            target_layers=[self.target_layer],
            reshape_transform=self.reshape_transform,
        )
        try:
            grayscale = cam(input_tensor=input_tensor, targets=targets)
        finally:
            if hasattr(cam, "__exit__"):
                cam.__exit__(None, None, None)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        heatmap = normalize_heatmap(np.asarray(grayscale[0]))
        return XAIResult(
            method=self.name,
            heatmap=heatmap,
            elapsed_ms=elapsed_ms,
            target_class=target_class,
        )


class GradCAM(_GradCAMBase):
    name = "Grad-CAM"
    cam_class = _GradCAM


class GradCAMpp(_GradCAMBase):
    name = "Grad-CAM++"
    cam_class = _GradCAMpp
