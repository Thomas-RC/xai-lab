from io import BytesIO

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224


_resize_crop = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(INPUT_SIZE),
    ]
)

_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

_to_tensor = transforms.ToTensor()


def load_image(source: str | bytes | BytesIO | Image.Image) -> Image.Image:
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, (bytes, BytesIO)):
        buf = BytesIO(source) if isinstance(source, bytes) else source
        return Image.open(buf).convert("RGB")
    return Image.open(source).convert("RGB")


def to_display_array(img: Image.Image) -> np.ndarray:
    """224x224 RGB w [0,1] float32 — do nakladania heatmapy."""
    cropped = _resize_crop(img)
    return np.asarray(cropped, dtype=np.float32) / 255.0


def to_model_tensor(img: Image.Image, device: str = "cpu") -> torch.Tensor:
    """Tensor [1,3,224,224] znormalizowany — wejscie do modelu."""
    cropped = _resize_crop(img)
    tensor = _to_tensor(cropped)
    tensor = _normalize(tensor)
    return tensor.unsqueeze(0).to(device)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Odwrotnosc normalizacji ImageNet — do wizualizacji tensora wstecz."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(tensor.device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(tensor.device)
    return (tensor * std + mean).clamp(0, 1)
