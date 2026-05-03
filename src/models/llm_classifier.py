"""Klasyfikator obrazow oparty na Gemini 2.5 Flash (Vision LLM via Vertex AI).

W przeciwienstwie do ResNet50/ViT-B/16 nie ma sztywnego slownika 1000 klas
ImageNet — zwraca free-form etykiety w jezyku polskim. Reprezentuje
"otwarty" klasyfikator do porownania ze "zamknietymi" modelami klasycznymi.

XAI klasyczne (Grad-CAM, IG, SmoothGrad) na tym modelu **nie zadzialaja** —
brak feature map / logitow klas.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field

from google import genai
from google.genai import types
from google.oauth2 import service_account
from PIL import Image
from pydantic import BaseModel, Field

_MODEL = "gemini-2.5-flash"
_PROMPT = """Analyze the image. Return ONLY valid JSON (no markdown, no comments)
with exactly this shape, with exactly 5 items in "predictions", sorted
descending by confidence:

{"predictions":[{"label":"class1","confidence":0.5},{"label":"class2","confidence":0.2},{"label":"class3","confidence":0.15},{"label":"class4","confidence":0.1},{"label":"class5","confidence":0.05}],"reasoning":"one or two sentences of reasoning"}

Requirements:
- class names in English (animal species, objects, scenes)
- confidence is a fraction [0,1], summing to <= 1.0
- reasoning: what you concretely see, which features led to this classification
"""

# Vertex AI w europe-west9 z multimodal nie wspiera pydantic-class jako
# response_schema (server zrywa polaczenie) — uzywamy plain dict JSON Schema.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "confidence"],
            },
        },
        "reasoning": {"type": "string"},
    },
    "required": ["predictions", "reasoning"],
}


class Prediction(BaseModel):
    label: str = Field(description="Nazwa klasy po polsku")
    confidence: float = Field(ge=0, le=1, description="Pewnosc, [0,1]")


class VisionResponse(BaseModel):
    predictions: list[Prediction] = Field(description="Top-5 klas, malejaco po pewnosci")
    reasoning: str = Field(description="1-2 zdania uzasadnienia")


@dataclass
class LLMClassifier:
    name: str = "gemini_vision"
    is_transformer: bool = False
    is_llm: bool = True
    _client: genai.Client | None = field(default=None, repr=False)

    def _get_client(self) -> genai.Client:
        if self._client is not None:
            return self._client

        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION")
        creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not project or not location:
            raise RuntimeError(
                "Brak GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION w env."
            )

        if creds_path and os.path.isfile(creds_path):
            creds = service_account.Credentials.from_service_account_file(
                creds_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"],
            )
            self._client = genai.Client(
                vertexai=True, project=project, location=location, credentials=creds
            )
        else:
            self._client = genai.Client(vertexai=True, project=project, location=location)
        return self._client

    def classify(self, image: Image.Image) -> VisionResponse:
        client = self._get_client()
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        image_part = types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png")

        resp = client.models.generate_content(
            model=_MODEL,
            contents=[image_part, _PROMPT],
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        # Vertex w europe-west9 sporadycznie dopisuje drugi JSON po pierwszym
        # (temp=0 ogranicza ale nie eliminuje) — bierzemy tylko pierwszy obiekt.
        raw, _ = json.JSONDecoder().raw_decode(resp.text.lstrip())
        if not isinstance(raw, dict):
            raise ValueError(f"Gemini zwrocil nieoczekiwany format: {resp.text!r}")
        return VisionResponse(**raw)
