"""Loader pretrained modeli ImageNet: ResNet50 (CNN) i ViT-B/16 (Transformer).

Plus LLMClassifier (Gemini 2.5 Flash via Vertex AI) jako trzeci, "otwarty"
klasyfikator do porownania ze zamknietymi modelami ImageNet.

Singletony cachowane w RAM — pierwszy load ~30 s (download + init), kolejne O(1).
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch
from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)

from src.models.clip_classifier import CLIPClassifier
from src.models.llm_classifier import LLMClassifier
from src.models.siglip_classifier import SigLIPClassifier

ModelName = Literal["resnet50", "vit_b_16", "clip", "siglip2", "gemini_vision"]


@dataclass
class LoadedModel:
    name: str
    model: nn.Module
    target_layer: nn.Module
    """Ostatnia warstwa konwolucyjna (CNN) lub ostatni blok normalizujacy (ViT) —
    do Grad-CAM. Dla ViT wymagany 'reshape_transform' przy uzyciu w pytorch-grad-cam."""
    is_transformer: bool
    is_llm: bool = False


@lru_cache(maxsize=5)
def load_model(
    name: ModelName = "resnet50", device: str = "cpu"
) -> LoadedModel | LLMClassifier | CLIPClassifier | SigLIPClassifier:
    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = resnet50(weights=weights)
        target_layer = model.layer4[-1]
        is_transformer = False
    elif name == "vit_b_16":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = vit_b_16(weights=weights)
        target_layer = model.encoder.layers[-1].ln_1
        is_transformer = True
    elif name == "gemini_vision":
        return LLMClassifier()
    elif name == "clip":
        return CLIPClassifier(device=device)
    elif name == "siglip2":
        return SigLIPClassifier(device=device)
    else:
        raise ValueError(f"Nieznany model: {name}")

    model.eval()
    model.to(device)
    # NIE zerujemy requires_grad parametrow — Grad-CAM potrzebuje gradientow
    # plynacych przez warstwy featurowe. eval() wystarczy by zablokowac BN/dropout drift.

    return LoadedModel(
        name=name,
        model=model,
        target_layer=target_layer,
        is_transformer=is_transformer,
    )


def vit_reshape_transform(tensor: torch.Tensor, height: int = 14, width: int = 14) -> torch.Tensor:
    """ViT-B/16 zwraca (B, 197, D): 1 token CLS + 196 patchy 14x14.
    Przeksztalcamy na (B, D, 14, 14) zgodnie z konwencja CNN dla pytorch-grad-cam.
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.permute(0, 3, 1, 2)


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Forward pass + softmax. Zwraca tensor [batch, 1000] prawdopodobieństw."""
    logits = model(x)
    return torch.softmax(logits, dim=-1)
