"""Etykiety ImageNet 1000-klasowe — pobierane z metadanych weights torchvision.

Używamy `ResNet50_Weights.IMAGENET1K_V2.meta["categories"]`, które torchvision
gwarantuje jako stabilne mapowanie idx → human-readable label. To samo
mapowanie obowiązuje dla wszystkich modeli pretrained na ImageNet-1k
(ResNet, ViT, EfficientNet, ...) bo używają tej samej kolejności klas.
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def imagenet_categories() -> list[str]:
    from torchvision.models import ResNet50_Weights

    return list(ResNet50_Weights.IMAGENET1K_V2.meta["categories"])


def label_for(idx: int) -> str:
    cats = imagenet_categories()
    if not 0 <= idx < len(cats):
        raise ValueError(f"idx {idx} poza zakresem [0, {len(cats)})")
    return cats[idx]


def top_k(probabilities, k: int = 5) -> list[tuple[int, str, float]]:
    """Zwraca [(idx, label, prob), ...] dla top-k klas."""
    import torch

    if not isinstance(probabilities, torch.Tensor):
        probabilities = torch.as_tensor(probabilities)
    if probabilities.dim() == 2:
        probabilities = probabilities[0]
    probs, indices = torch.topk(probabilities, k=k)
    return [
        (int(idx), label_for(int(idx)), float(prob))
        for idx, prob in zip(indices.tolist(), probs.tolist(), strict=True)
    ]
