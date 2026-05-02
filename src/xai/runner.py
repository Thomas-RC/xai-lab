"""Wspolny entry point: uruchom wszystkie metody XAI dla (model, image, target)."""

from dataclasses import dataclass

import torch

from src.config import settings
from src.models.loader import LoadedModel, vit_reshape_transform
from src.xai import (
    GradCAM,
    GradCAMpp,
    IntegratedGradients,
    LimeImage,
    Occlusion,
    SmoothGrad,
    XAIResult,
)


@dataclass
class XAIBundle:
    target_class: int
    results: list[XAIResult]


def build_methods(loaded: LoadedModel) -> list:
    reshape = vit_reshape_transform if loaded.is_transformer else None
    return [
        GradCAM(target_layer=loaded.target_layer, reshape_transform=reshape),
        GradCAMpp(target_layer=loaded.target_layer, reshape_transform=reshape),
        IntegratedGradients(n_steps=settings.ig_steps),
        SmoothGrad(n_samples=settings.smoothgrad_samples, sigma=settings.smoothgrad_sigma),
        Occlusion(patch=settings.occlusion_patch, stride=settings.occlusion_stride),
        LimeImage(num_samples=settings.lime_samples, num_segments=settings.lime_segments),
    ]


def run_all(
    loaded: LoadedModel,
    input_tensor: torch.Tensor,
    target_class: int,
) -> XAIBundle:
    methods = build_methods(loaded)
    results: list[XAIResult] = []
    for method in methods:
        try:
            result = method.explain(loaded.model, input_tensor, target_class)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            results.append(
                XAIResult(
                    method=method.name,
                    heatmap=__import__("numpy").zeros((224, 224), dtype="float32"),
                    elapsed_ms=0.0,
                    target_class=target_class,
                    extra={"error": str(exc)},
                )
            )
    return XAIBundle(target_class=target_class, results=results)
