# xai-lab

Webowa stacja diagnostyczna dla klasyfikatorów obrazów. Uploadujesz zdjęcie,
wybierasz model (ResNet50 / ViT-B/16 / Gemini 2.5 Flash Vision), klikasz
**Wyjaśnij** — i dostajesz **6 heatmap obok siebie** (dla CNN/ViT) albo
strukturalną odpowiedź LLM z uzasadnieniem (dla Gemini Vision).
Każda heatmapa wygenerowana inną metodą XAI:

| # | Metoda | Rok | Typ |
|---|---|---|---|
| 1 | Grad-CAM | 2017 | gradient × aktywacje (Selvaraju) |
| 2 | Grad-CAM++ | 2018 | wyższe momenty gradientu |
| 3 | Integrated Gradients | 2017 | aksjomatyczna ścieżkowa (Sundararajan) |
| 4 | SmoothGrad | 2017 | uśrednianie zaszumionych kopii (Smilkov) |
| 5 | Occlusion | 2014 | model-agnostic, zasłanianie regionów (Zeiler) |
| 6 | LIME | 2016 | lokalny surrogate na super-pikselach (Ribeiro) |

Projekt zaliczeniowy "Zaawansowane metody sztucznej inteligencji" — próg 5.0,
rozszerzenie rozdziału 10 z F. Cholleta *Deep Learning with Python*.

## Quick start

```bash
cp .env.example .env
# Wrzuć GCP service account JSON do ./secrets/, zaktualizuj GCP_* w .env
make up                 # → http://xai.local.pl
make logs               # podgląd logów
```

## Struktura repo

Patrz [PLAN.md](PLAN.md) — pełny plan, architektura, decyzje projektowe.

## Stack

- Python 3.12, PyTorch 2.5 (CUDA 12.4), torchvision
- Captum + pytorch-grad-cam + LIME
- Streamlit (UI po polsku), WeasyPrint (Markdown → PDF)
- Docker + Traefik, GPU passthrough (nvidia-container-toolkit)
- Gemini 2.5 Flash via Vertex AI (`google-genai`) — dwie role:
  (1) tłumaczenie etykiet ImageNet EN→PL on-demand,
  (2) trzeci klasyfikator (Vision LLM, otwarty słownik) do porównania
  z ResNet/ViT — region `europe-west9` / EOG

## Licencja

Kod: [MIT](LICENSE) © 2026 Thomas-RC.
Modele i dane: licencje upstream (torchvision / ImageNet).
