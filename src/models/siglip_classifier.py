"""SigLIP 2 zero-shot classifier (Google 2025, via HuggingFace transformers).

SigLIP 2 (arXiv:2502.14786, luty 2025) to nastepca CLIP/SigLIP od Google'a —
ten sam paradygmat (encoder obrazu + encoder tekstu, NIE LLM) ale lepiej
trenowany (multilingual, wiecej danych, nowsze loss). Open-vocabulary
zero-shot jak CLIP, otwarte wagi (Apache 2.0).

Uzywany jako 5. klasyfikator w `compare_classifiers.py` — pokazuje ze
non-LLM SOTA z 2025 dorownuje Geminiemu Vision na otwartym slowniku.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import torch
from PIL import Image

# Te same kandydaty co CLIP — porownywalnie z compare_classifiers.py
from src.models.clip_classifier import _build_candidates


_MODEL_ID = "google/siglip2-base-patch16-224"


@lru_cache(maxsize=1)
def _load_siglip(device: str = "cpu"):
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(_MODEL_ID)
    model = AutoModel.from_pretrained(_MODEL_ID).to(device).eval()
    return model, processor


@dataclass
class SigLIPPrediction:
    label: str
    confidence: float


@dataclass
class SigLIPResponse:
    predictions: list[SigLIPPrediction]


@dataclass
class SigLIPClassifier:
    name: str = "siglip2_zero_shot"
    is_transformer: bool = True
    is_llm: bool = False
    device: str = "cpu"
    _candidates: list[str] = field(default_factory=_build_candidates)

    @torch.inference_mode()
    def classify(self, image: Image.Image, top_k: int = 5) -> SigLIPResponse:
        model, processor = _load_siglip(self.device)

        # SigLIP 2 wymaga prefixu "This is a photo of {label}." dla zero-shot
        # (zgodnie z model card na HuggingFace)
        prompts = [f"This is a photo of {c}." for c in self._candidates]

        inputs = processor(
            text=prompts,
            images=image.convert("RGB"),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = model(**inputs)
        logits = outputs.logits_per_image.squeeze(0)
        # SigLIP uses sigmoid loss — interpretujemy logity jako pewnosc per klase,
        # ale do top-k uzyjemy softmax dla porownywalnosci z CLIP
        probs = logits.softmax(dim=-1)

        top_probs, top_idx = probs.topk(top_k)
        predictions = [
            SigLIPPrediction(label=self._candidates[int(i)], confidence=float(p))
            for i, p in zip(top_idx.tolist(), top_probs.tolist(), strict=True)
        ]
        return SigLIPResponse(predictions=predictions)
