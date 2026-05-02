"""Smoke testy ladowania modeli i klasyfikacji.

Uwaga: pierwszy test pobiera wagi (~100 MB ResNet, ~330 MB ViT) z hub torchvision.
Cache w `~/.cache/torch/hub/checkpoints/` — kolejne uruchomienia natychmiastowe.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.models.loader import load_model, predict, vit_reshape_transform
from src.utils.imagenet import imagenet_categories, label_for, top_k
from src.utils.preprocess import to_display_array, to_model_tensor


@pytest.fixture
def sample_image() -> Image.Image:
    """Sztuczny obraz - gradient kolorow. Klasyfikacja niewazna, tylko ksztalt."""
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    for i in range(300):
        arr[i, :, 0] = i * 255 // 300
        arr[:, i, 1] = i * 255 // 300
    return Image.fromarray(arr)


def test_imagenet_categories_count():
    cats = imagenet_categories()
    assert len(cats) == 1000
    assert cats[0] == "tench"
    assert cats[207] == "golden retriever"


def test_label_for_valid():
    assert "retriever" in label_for(207)


def test_label_for_invalid():
    with pytest.raises(ValueError):
        label_for(1500)


def test_preprocess_shapes(sample_image):
    display = to_display_array(sample_image)
    tensor = to_model_tensor(sample_image)
    assert display.shape == (224, 224, 3)
    assert display.dtype == np.float32
    assert 0.0 <= display.min() and display.max() <= 1.0
    assert tensor.shape == (1, 3, 224, 224)


@pytest.mark.slow
def test_resnet_load_and_predict(sample_image):
    loaded = load_model("resnet50")
    assert loaded.name == "resnet50"
    assert loaded.is_transformer is False

    x = to_model_tensor(sample_image)
    probs = predict(loaded.model, x)
    assert probs.shape == (1, 1000)
    assert torch.allclose(probs.sum(), torch.tensor(1.0), atol=1e-4)

    top = top_k(probs, k=3)
    assert len(top) == 3
    assert all(0 <= idx < 1000 for idx, _, _ in top)


@pytest.mark.slow
def test_vit_load_and_predict(sample_image):
    loaded = load_model("vit_b_16")
    assert loaded.is_transformer is True

    x = to_model_tensor(sample_image)
    probs = predict(loaded.model, x)
    assert probs.shape == (1, 1000)


def test_vit_reshape_transform():
    fake = torch.randn(1, 197, 768)
    out = vit_reshape_transform(fake)
    assert out.shape == (1, 768, 14, 14)
