"""Tlumaczenie etykiet ImageNet EN -> PL przez Gemini 2.5 Flash (Vertex AI).

Batch — caly top-5 idzie w 1 request (taniej niz 5 osobnych).
Cache w pamieci procesu — re-translate tych samych etykiet nie bije Vertexa.

Zmienne srodowiskowe (ustawiane w docker-compose.yml):
- GOOGLE_APPLICATION_CREDENTIALS — sciezka do SA JSON (./secrets/...)
- GOOGLE_CLOUD_PROJECT
- GOOGLE_CLOUD_LOCATION
- GOOGLE_GENAI_USE_VERTEXAI=True
"""

from __future__ import annotations

import json
import os

from google import genai
from google.genai import types
from google.oauth2 import service_account

_MODEL = "gemini-2.5-flash"
_PROMPT = """Przetlumacz nizsze nazwy klas ImageNet z angielskiego na polski.
Wymagania:
- pojedyncze slowo lub krotka fraza (max 4 wyrazy)
- dla zwierzat — polskie nazwy gatunkow
- dla obiektow — utarte polskie odpowiedniki (np. "samolot odrzutowy")
- jesli brak naturalnego polskiego odpowiednika, zostaw oryginal angielski
- pierwsza litera mala (chyba ze nazwa wlasna)

Wejscie (JSON list): {labels}

Odpowiedz TYLKO JSONem — lista stringow, ta sama dlugosc i kolejnosc co wejscie.
"""

_client: genai.Client | None = None
_cache: dict[str, str] = {}


def _get_client() -> genai.Client:
    global _client
    if _client is not None:
        return _client

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION")
    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not project or not location:
        raise RuntimeError(
            "Brak GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION w env. "
            "Sprawdz .env i docker-compose.yml."
        )

    if creds_path and os.path.isfile(creds_path):
        creds = service_account.Credentials.from_service_account_file(
            creds_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        _client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            credentials=creds,
        )
    else:
        _client = genai.Client(vertexai=True, project=project, location=location)

    return _client


def _call_gemini(labels: list[str]) -> list[str]:
    client = _get_client()
    prompt = _PROMPT.format(labels=json.dumps(labels, ensure_ascii=False))
    resp = client.models.generate_content(
        model=_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            response_schema={"type": "array", "items": {"type": "string"}},
        ),
    )
    parsed = resp.parsed
    if parsed is None:
        parsed = json.loads(resp.text)
    if not isinstance(parsed, list) or len(parsed) != len(labels):
        raise ValueError(
            f"Gemini zwrocil nieoczekiwany format (expected list[{len(labels)}]): "
            f"{resp.text!r}"
        )
    return [str(s) for s in parsed]


def translate_labels(labels: list[str]) -> list[str]:
    """EN -> PL przez Gemini 2.5 Flash. Batch w 1 requeście, cache per proces."""
    if not labels:
        return []
    missing = [lbl for lbl in labels if lbl not in _cache]
    if missing:
        unique_missing = list(dict.fromkeys(missing))
        translated = _call_gemini(unique_missing)
        for src, pl in zip(unique_missing, translated, strict=True):
            _cache[src] = pl
    return [_cache[lbl] for lbl in labels]
