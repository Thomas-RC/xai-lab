"""LIME (Ribeiro 2016) dla obrazow.

LIME segmentuje obraz na super-piksele (SLIC), generuje N perturbacji
(losowo wlaczane/wylaczane segmenty), klasyfikuje kazda i fituje regresje
liniowa wagujaca segmenty wedlug wplywu na prawdopodobienstwo klasy.

Wynik: wagi per super-piksel — przeksztalcane na heatmape per piksel.
"""

import time

import numpy as np
import torch
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic
from torch import nn

from src.utils.preprocess import IMAGENET_MEAN, IMAGENET_STD
from src.xai.base import XAIMethod, XAIResult, normalize_heatmap


def _make_predict_fn(model: nn.Module, device: str):
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 1, 3)

    @torch.inference_mode()
    def predict_fn(images: np.ndarray) -> np.ndarray:
        # images: [N, H, W, 3] uint8 lub [0,1] float — LIME daje uint8
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        normalized = (images - mean) / std
        tensor = torch.from_numpy(normalized).permute(0, 3, 1, 2).contiguous().float()
        tensor = tensor.to(device)
        probs = torch.softmax(model(tensor), dim=-1)
        return probs.cpu().numpy()

    return predict_fn


class LimeImage(XAIMethod):
    name = "LIME"

    def __init__(self, num_samples: int = 300, num_segments: int = 50):
        self.num_samples = num_samples
        self.num_segments = num_segments

    def explain(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> XAIResult:
        device = str(input_tensor.device)

        # Odwrocenie normalizacji ImageNet do RGB uint8 (wejscie LIME)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).to(input_tensor.device)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1).to(input_tensor.device)
        rgb_float = (input_tensor[0] * std + mean).clamp(0, 1)
        rgb_uint8 = (rgb_float.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        explainer = LimeImageExplainer()
        predict_fn = _make_predict_fn(model, device)

        def segmentation_fn(image: np.ndarray) -> np.ndarray:
            return slic(image, n_segments=self.num_segments, compactness=10, sigma=1, start_label=0)

        t0 = time.perf_counter()
        explanation = explainer.explain_instance(
            image=rgb_uint8,
            classifier_fn=predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=self.num_samples,
            segmentation_fn=segmentation_fn,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Mapuj wagi segmentow na piksele
        segments = explanation.segments
        weights = dict(explanation.local_exp.get(target_class, []))
        if not weights:
            # Fallback: top label LIME (gdy target rozni sie od top-1)
            top = explanation.top_labels[0]
            weights = dict(explanation.local_exp[top])

        heatmap = np.zeros(segments.shape, dtype=np.float32)
        for seg_id, weight in weights.items():
            heatmap[segments == seg_id] = weight

        return XAIResult(
            method=self.name,
            heatmap=normalize_heatmap(heatmap),
            elapsed_ms=elapsed_ms,
            target_class=target_class,
            extra={"num_samples": self.num_samples, "num_segments": self.num_segments},
        )
