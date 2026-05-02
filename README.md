# xai-lab

Webowa stacja diagnostyczna dla klasyfikatorów obrazów. Uploadujesz zdjęcie,
wybierasz model (ResNet50 / ViT-B/16), klikasz **Wyjaśnij** — i dostajesz
**6 heatmap obok siebie**, każda wygenerowana inną metodą XAI:

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
make up                 # → http://xai.local.pl
make logs               # podgląd logów
```

## Generowanie sprawozdania PDF

```bash
make report             # generuje report/report.pdf z figurami
```

## Struktura repo

Patrz [PLAN.md](PLAN.md) — pełny plan, architektura, decyzje projektowe.

```
xai-lab/
├── PLAN.md                     # plan projektu
├── docker-compose.yml
├── pyproject.toml
├── deploy/Dockerfile.app
├── src/
│   ├── models/loader.py        # ResNet50 + ViT-B/16
│   ├── xai/                    # 6 metod XAI
│   ├── ui/streamlit_app.py     # web UI
│   ├── metrics/                # IoU, timing
│   └── utils/                  # preprocess, ImageNet (PL+EN), viz
├── data/samples/               # 9 obrazów testowych
├── scripts/{run_batch,smoke_app}.py
├── tests/
└── report/{report.md, build_pdf.py, figures/}
```

## Stack

- Python 3.12, PyTorch 2.5 (CUDA 12.4), torchvision
- Captum + pytorch-grad-cam + LIME
- Streamlit (UI po polsku), WeasyPrint (Markdown → PDF)
- Docker + Traefik, GPU passthrough (nvidia-container-toolkit)

## Licencja

Projekt edukacyjny. Modele i dane: licencje upstream (torchvision / ImageNet).
