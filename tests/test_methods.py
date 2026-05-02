"""Testy kontraktow metod XAI: kazda metoda zwraca [H, W] in [0, 1] o niezerowym sygnale.

Uzywamy malego modelu zastepczego (a nie ResNet50) zeby testy byly szybkie.
ResNet50 + ViT testowane sa w `tests/test_models.py` osobno.
"""

import numpy as np
import pytest
import torch
from torch import nn

from src.xai import (
    GradCAM,
    GradCAMpp,
    IntegratedGradients,
    LimeImage,
    Occlusion,
    SmoothGrad,
)


class TinyCNN(nn.Module):
    """Minimalna siec konwolucyjna do testow XAI: input [1,3,224,224] -> 10 klas."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        h = self.features(x)
        return self.classifier(h.flatten(1))


@pytest.fixture
def model() -> TinyCNN:
    torch.manual_seed(42)
    m = TinyCNN()
    m.eval()
    return m


@pytest.fixture
def input_tensor() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(1, 3, 224, 224, requires_grad=True)


def _check_heatmap(heatmap: np.ndarray):
    assert heatmap.ndim == 2
    assert heatmap.dtype == np.float32
    assert heatmap.shape[0] > 0 and heatmap.shape[1] > 0
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-5
    assert heatmap.max() > 0.0, "heatmap pusta — metoda nic nie wykryla"


def test_gradcam(model, input_tensor):
    method = GradCAM(target_layer=model.features[-2])
    result = method.explain(model, input_tensor, target_class=0)
    assert result.method == "Grad-CAM"
    _check_heatmap(result.heatmap)


def test_gradcam_pp(model, input_tensor):
    method = GradCAMpp(target_layer=model.features[-2])
    result = method.explain(model, input_tensor, target_class=0)
    assert result.method == "Grad-CAM++"
    _check_heatmap(result.heatmap)


def test_integrated_gradients(model, input_tensor):
    method = IntegratedGradients(n_steps=10)
    result = method.explain(model, input_tensor, target_class=0)
    _check_heatmap(result.heatmap)
    assert result.heatmap.shape == (224, 224)


def test_smoothgrad(model, input_tensor):
    method = SmoothGrad(n_samples=5, sigma=0.15)
    result = method.explain(model, input_tensor, target_class=0)
    _check_heatmap(result.heatmap)
    assert result.heatmap.shape == (224, 224)


def test_occlusion(model, input_tensor):
    method = Occlusion(patch=64, stride=64)
    result = method.explain(model, input_tensor.detach(), target_class=0)
    _check_heatmap(result.heatmap)
    assert result.heatmap.shape == (224, 224)


@pytest.mark.slow
def test_lime(model, input_tensor):
    method = LimeImage(num_samples=50, num_segments=20)
    result = method.explain(model, input_tensor.detach(), target_class=0)
    _check_heatmap(result.heatmap)
