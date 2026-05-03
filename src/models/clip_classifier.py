"""CLIP zero-shot classifier (OpenAI 2021, via open-clip-torch).

Encoder obrazu (ViT) + encoder tekstu — bez generowania tokenow, bez LLM.
W runtime dostaje liste kandydatow tekstowych, zwraca top-5 po cosine
similarity miedzy embeddingiem obrazu a embeddingami kazdego kandydata.

Uzywany jako 4. klasyfikator w `compare_classifiers.py` — pokazuje ze
otwarto-slownikowa klasyfikacja **nie wymaga LLM-a** (CLIP to czyste
dwa transformery, gradient policzalny → klasyczne XAI by zadzialalo).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import torch
from PIL import Image

# Domyslna lista kandydatow: 1000 etykiet ImageNet (skrocone) + 20 dodanych
# recznie aby pokryc obrazy spoza ImageNet (galaktyka, burza, Concorde, zbik).
_EXTRA_CANDIDATES = [
    "spiral galaxy",
    "Milky Way galaxy",
    "lightning storm over a field",
    "Concorde supersonic airliner",
    "European wildcat (Felis silvestris)",
    "domestic cat",
    "lynx",
    "tabby cat",
    "person holding a small animal in a towel",
    "person feeding a cat",
    "European brown bear with cubs",
    "scorpion on red sand",
    "platypus held in a towel",
    "Porsche 911 sports car",
    "small bird singing on a branch",
    "thunderstorm and lightning",
    "river beaver in winter",
    "starry night sky",
    "wild animal in nature",
    "everyday object",
]


@lru_cache(maxsize=1)
def _load_clip(device: str = "cpu"):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.to(device).eval()
    return model, preprocess, tokenizer


@lru_cache(maxsize=1)
def _imagenet_labels() -> tuple[str, ...]:
    from torchvision.models import ResNet50_Weights

    cats = ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    return tuple(cats)


def _build_candidates() -> list[str]:
    return list(_imagenet_labels()) + _EXTRA_CANDIDATES


@dataclass
class CLIPPrediction:
    label: str
    confidence: float


@dataclass
class CLIPResponse:
    predictions: list[CLIPPrediction]


@dataclass
class CLIPClassifier:
    name: str = "clip_zero_shot"
    is_transformer: bool = True
    is_llm: bool = False
    device: str = "cpu"
    _candidates: list[str] = field(default_factory=_build_candidates)

    @torch.inference_mode()
    def classify(self, image: Image.Image, top_k: int = 5) -> CLIPResponse:
        model, preprocess, tokenizer = _load_clip(self.device)
        x = preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        text = tokenizer(self._candidates).to(self.device)

        img_emb = model.encode_image(x)
        txt_emb = model.encode_text(text)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

        # Logit scale dla normalizacji do prawdopodobienstw
        logit_scale = model.logit_scale.exp()
        logits = (logit_scale * img_emb @ txt_emb.T).squeeze(0)
        probs = logits.softmax(dim=-1)

        top_probs, top_idx = probs.topk(top_k)
        predictions = [
            CLIPPrediction(label=self._candidates[int(i)], confidence=float(p))
            for i, p in zip(top_idx.tolist(), top_probs.tolist(), strict=True)
        ]
        return CLIPResponse(predictions=predictions)
