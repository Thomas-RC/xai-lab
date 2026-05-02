# Plan wdrożenia: xai-lab

Webowa stacja diagnostyczna dla klasyfikatorów obrazów. Użytkownik wgrywa zdjęcie,
wybiera model (CNN lub ViT) i otrzymuje **6 heatmap obok siebie** — każda
wygenerowana inną metodą XAI — wraz z porównaniem ilościowym i jakościowym.

Projekt zaliczeniowy "Zaawansowane metody sztucznej inteligencji" — próg **5.0**,
rozszerzenie rozdziału 10 z książki F. Cholleta *Deep Learning with Python*
(wyd. 2 / online edition: deeplearningwithpython.io/chapters/chapter10).

---

## 1. Cel i zakres

**Cel dydaktyczny:** wykazać, że Grad-CAM (jedyna metoda XAI prezentowana
w rozdziale 10 Cholleta) jest **jedną z wielu** technik, każda o innych
założeniach matematycznych, ograniczeniach i obrazie "prawdy" o decyzji modelu.

**Cel praktyczny:** narzędzie do auditu klasyfikatorów obrazów przed wdrożeniem
— wykrywanie shortcut learning (model patrzy w złe miejsce), weryfikacja
zgodności z RODO / EU AI Act (right to explanation), debug błędnych predykcji.

**Co robi aplikacja:**

- Upload obrazu (drag & drop) → klasyfikacja na ImageNet (1000 klas) z top-5
  prawdopodobieństw
- Generacja **6 heatmap** dla wybranej klasy (domyślnie top-1) —
  każda inną metodą XAI, ten sam model
- Galeria porównawcza obok siebie + nakładka heatmapy na obraz
- **Tabela ilościowa**: czas inferencji, rozdzielczość, IoU heatmap
  parami (zgodność metod między sobą)
- Eksport: ZIP z heatmapami PNG + JSON z metadanymi (do sprawozdania)

**Co poza zakresem (świadomie):**

- Trening własnego modelu — używamy pretrained (oszczędność czasu, koncentracja
  na XAI a nie na klasyfikacji)
- Detekcja obiektów / segmentacja — tylko klasyfikacja (zgodnie z rozdziałem 10)
- Tekstowe XAI (NLP) — tylko wizja
- TCAV i counterfactuals jako ścieżki rozwoju (rozdz. 9), nie MVP

---

## 2. Stack

| Warstwa | Komponent | Wersja | Uzasadnienie |
|---|---|---|---|
| Język | Python | 3.12 | spójność z drugą apką, wsparcie torch 2.x |
| ML framework | PyTorch + torchvision | 2.5+ | ekosystem XAI (Captum) żyje w PyTorch, nie Keras |
| Modele CNN | ResNet50 (ImageNet) | torchvision | klasyk, dobrze zbadany, działa Grad-CAM |
| Modele ViT | ViT-B/16 (ImageNet) | torchvision | Grad-CAM wymaga `reshape_transform` — kontrast |
| GPU | NVIDIA RTX (CUDA 12.4) | nvidia-container-toolkit | torch CUDA, ~10× szybsze niż CPU |
| XAI: gradientowe | Captum | 0.7+ | Integrated Gradients, SmoothGrad, Saliency, GradientShap |
| XAI: Grad-CAM | pytorch-grad-cam (jacobgil) | 1.5+ | Grad-CAM, Grad-CAM++, EigenCAM |
| XAI: surrogate | LIME (marcotcr/lime) | 0.2+ | model-agnostic, super-piksele |
| XAI: occlusion | własna implementacja | — | dydaktycznie warto napisać samemu (Zeiler 2014) |
| Web UI | Streamlit | 1.40+ | minimum tarcia, gotowe upload + wykresy |
| Reverse proxy | Traefik (zewnętrzna sieć `proxy`) | — | spójność z konwencją `*.local.pl` |
| Sprawozdanie | Markdown → WeasyPrint → PDF | — | bez LaTeX, lekka zależność |
| CI lokalne | Makefile + pytest + ruff | — | spójność z drugą apką |

**Decyzje świadomie odrzucone:**

- **Keras / TensorFlow** (jak w książce) → PyTorch. Powód: ekosystem XAI
  (Captum, pytorch-grad-cam, LIME) jest dojrzały tylko w torchu. Większa
  różnica względem książki = mocniejszy argument na 5.0.
- **CPU-only** → ostatecznie torch CUDA + GPU passthrough. CPU 6 metod
  na obrazie zajmuje ~30 s (LIME ~8 s sam), GPU (RTX 4070): ResNet50
  cały zestaw ~10 s, ViT-B/16 ~2 min (model 3× większy + 750 forward'ów
  dla LIME/Occlusion/IG/SG łącznie). UI wykrywa CUDA i pokazuje opcję.
- **Trening custom modelu** → pretrained ImageNet. Skupiamy budżet czasu
  na XAI, nie na uczeniu od zera (które i tak rozdz. 10 nie wymaga).
- **MLflow / W&B do trackingu eksperymentów** → niepotrzebne, nie ma
  treningu, jedna konfiguracja.
- **TCAV (Concept Activation Vectors)** → ścieżka rozwoju. Wymaga
  zbioru obrazów koncepcyjnych (np. "paski") — narzut, którego MVP
  nie udźwignie.
- **pandoc + LaTeX do PDF** → WeasyPrint. LaTeX waży gigabajty
  w obrazie Dockera, WeasyPrint to ~50 MB.

---

## 3. Architektura

```
┌──────────────────────────────────────────────────────────────────┐
│                    docker-compose network: proxy (external)      │
│                                                                  │
│              ┌────────────────────┐                              │
│              │   Traefik (host)   │  routing po Host header      │
│              │   xai.local.pl     │                              │
│              └─────────┬──────────┘                              │
│                        │                                          │
│                        ▼                                          │
│            ┌──────────────────────────┐                          │
│            │         app              │                          │
│            │   Streamlit (port 8501)  │                          │
│            │                          │                          │
│            │   ┌────────────────────┐ │                          │
│            │   │  Model loader      │ │  ResNet50 + ViT-B/16     │
│            │   │  (torchvision)     │ │  pretrained, RAM-cached  │
│            │   └─────────┬──────────┘ │                          │
│            │             │            │                          │
│            │   ┌─────────▼──────────┐ │                          │
│            │   │  XAI engine        │ │  6 metod równolegle      │
│            │   │  ├─ GradCAM        │ │  (lub sekwencyjnie       │
│            │   │  ├─ GradCAM++      │ │   przy CPU)              │
│            │   │  ├─ IntegratedGrad │ │                          │
│            │   │  ├─ SmoothGrad     │ │                          │
│            │   │  ├─ Occlusion      │ │                          │
│            │   │  └─ LIME           │ │                          │
│            │   └─────────┬──────────┘ │                          │
│            │             │            │                          │
│            │   ┌─────────▼──────────┐ │                          │
│            │   │  Visualizer        │ │  matplotlib + PIL        │
│            │   │  (heatmap overlay) │ │                          │
│            │   └────────────────────┘ │                          │
│            └──────────────────────────┘                          │
│                                                                  │
│            ┌──────────────────────────┐                          │
│            │       report-gen         │  profil: report          │
│            │   (one-shot CLI)         │  generuje raport         │
│            │   scripts/run_batch.py   │  PDF na próbce obrazów   │
│            └──────────────────────────┘                          │
└──────────────────────────────────────────────────────────────────┘
```

### Przepływ pojedynczego zapytania

```
User upload image (UI)
       │
       ▼
Preprocess: resize 224×224, normalize (ImageNet stats)
       │
       ▼
Forward pass → top-5 predykcje (softmax)
       │
       ├─→ GradCAM     ── target_layer: layer4[-1] (ResNet) / blocks[-1].norm1 (ViT*)
       ├─→ GradCAM++   ── jw.
       ├─→ IntegratedGradients ── baseline: czarny obraz, 50 kroków
       ├─→ SmoothGrad  ── 50 sampli, σ=0.15
       ├─→ Occlusion   ── patch 32×32, stride 16, baseline szary
       └─→ LIME        ── 1000 sampli, segmenter SLIC (50 super-pikseli)
       │
       ▼
Heatmap[6] → upsample do 224×224 → normalize [0,1]
       │
       ▼
Overlay (jet colormap, α=0.5) + galeria 2×3
       │
       ▼
Metryki: czas, IoU parami, top-K piksele zgodne
       │
       ▼
Render w Streamlit + przycisk "Download ZIP"
```

\*ViT wymaga `reshape_transform` w pytorch-grad-cam — token CLS pomijany,
patche 14×14 → mapa 14×14 → upsample. Jeden z punktów dyskusji w sprawozdaniu.

---

## 4. Metody XAI — tabela porównawcza

| Metoda | Rok | Typ | Wymaga gradientów | Działa na ViT | Rozdzielczość | Złożoność |
|---|---|---|---|---|---|---|
| Grad-CAM | 2017 | gradient × aktywacje | tak | tak (z reshape) | niska (~14²) | 1× forward + 1× backward |
| Grad-CAM++ | 2018 | wyższe momenty | tak | tak | niska | 1× forward + 2× backward |
| Integrated Gradients | 2017 | aksjomatyczne (path) | tak | tak | pixel-level | N× forward + backward (N=50) |
| SmoothGrad | 2017 | szum + uśrednienie | tak | tak | pixel-level | N× IG (N=50) |
| Occlusion | 2014 | model-agnostic | nie | tak | średnia (patch) | (H/s × W/s) × forward |
| LIME | 2016 | lokalny surrogate | nie | tak | super-piksele | N× forward (N=1000) |

**Klucz do interpretacji** (treść sprawozdania):

- **Grad-CAM**: pokazuje "gdzie model patrzy" w sensie kanałów feature map
  ostatniej warstwy konwolucyjnej. Niska rozdzielczość = duże plamy.
- **Integrated Gradients**: matematycznie ścisła (spełnia aksjomaty Sensitivity
  i Implementation Invariance) — ale wynikiem jest pixel-level "noise-like" mapa,
  trudniejsza w interpretacji wizualnej.
- **SmoothGrad**: czyści szum z IG przez uśrednianie z wielu zaszumionych kopii.
- **Occlusion**: brutalnie prosty — zasłaniamy fragmenty i patrzymy, jak spada
  pewność. Jedyna metoda nie wymagająca dostępu do gradientów (czarna skrzynka OK).
- **LIME**: tłumaczy w przestrzeni super-pikseli (semantycznie zwartych regionów)
  — najbliższe ludzkiemu myśleniu, ale wolne i niedeterministyczne.

---

## 5. Layout repozytorium

```
xai-lab/
├── PLAN.md                       # ten plik
├── README.md                     # quick start: make up → xai.local.pl
├── Makefile                      # up / down / logs / test / report / fmt
├── docker-compose.yml            # app + (opcjonalnie) report-gen
├── pyproject.toml                # torch, torchvision, captum, lime, streamlit, weasyprint
├── .env.example                  # APP_PORT, MODEL_CACHE_DIR, DEFAULT_MODEL, DEVICE
├── .gitignore
├── deploy/
│   └── Dockerfile.app            # python:3.12-slim + torch CPU + captum
├── src/
│   ├── __init__.py
│   ├── config.py                 # pydantic-settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── loader.py             # ResNet50, ViT-B/16, cache w RAM
│   ├── xai/
│   │   ├── __init__.py
│   │   ├── base.py               # ABC: XAIMethod (input → heatmap [H,W] in [0,1])
│   │   ├── gradcam.py            # wrapper na pytorch-grad-cam
│   │   ├── gradcam_pp.py
│   │   ├── integrated_grads.py   # captum.IntegratedGradients
│   │   ├── smoothgrad.py         # captum.NoiseTunnel
│   │   ├── occlusion.py          # własna implementacja (~50 linii)
│   │   └── lime_xai.py           # lime.lime_image
│   ├── ui/
│   │   ├── __init__.py
│   │   └── streamlit_app.py      # entry point: streamlit run
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── iou.py                # IoU heatmap parami (po thresholdzie)
│   │   └── timing.py             # context manager dla pomiarów
│   └── utils/
│       ├── __init__.py
│       ├── imagenet.py           # mapping idx → label
│       ├── preprocess.py         # transforms ImageNet
│       └── viz.py                # overlay, gallery, colormap
├── data/
│   └── samples/                  # 8–10 obrazów .jpg (commit do repo)
│       ├── shepherd_with_ball.jpg
│       ├── cat_on_keyboard.jpg
│       ├── elephant_savanna.jpg
│       ├── airplane_sky.jpg
│       ├── espresso_cup.jpg
│       ├── pizza_pepperoni.jpg
│       ├── tiger_jungle.jpg
│       └── snowy_wolf.jpg        # nawiązanie do papera LIME
├── scripts/
│   ├── __init__.py
│   ├── run_batch.py              # generuje figury PNG dla sprawozdania
│   └── smoke_app.py              # health-check: 1 obraz, 6 metod, brak crashy
├── tests/
│   ├── __init__.py
│   ├── test_methods.py           # każda metoda zwraca [H,W] in [0,1]
│   ├── test_models.py            # ResNet i ViT ładują się i klasyfikują
│   └── test_metrics.py           # IoU symetryczne, w [0,1]
└── report/
    ├── report.md                 # treść sprawozdania
    ├── figures/                  # generowane przez scripts/run_batch.py
    └── build_pdf.py              # weasyprint: report.md → report.pdf
```

---

## 6. Plan rozwoju — status końcowy

Wszystkie batche **zakończone**:

- [x] **Batch 0** — szkielet (PLAN, README, pyproject, Makefile, .env.example, .gitignore)
- [x] **Batch 1** — config + modele (ResNet50, ViT-B/16, preprocess, ImageNet labels)
- [x] **Batch 2** — 6 metod XAI (Grad-CAM, GC++, IG, SmoothGrad, Occlusion, LIME)
- [x] **Batch 3** — wizualizacja + metryki (overlay, gallery 2×3, IoU, timing)
- [x] **Batch 4** — Streamlit UI (upload, klasyfikacja, heatmapy, ZIP download)
- [x] **Batch 5** — Docker + Traefik (`make up` → `http://xai.local.pl`)
- [x] **Batch 6** — sprawozdanie (`scripts/run_batch.py`, `report.md`, `build_pdf.py`)
- [x] **Batch 7** — szlif: 20/20 testów zielonych, ruff, smoke, GPU passthrough

**Dodatkowo (poza pierwotnym planem):**

- [x] **Polonizacja UI** — wszystkie napisy z ogonkami + słownik PL dla
      ImageNet (`src/utils/imagenet_pl.py`, ~200 klas + fallback do EN)
- [x] **Auto-detekcja CUDA** — UI pokazuje opcję GPU gdy dostępne; `run_batch.py`
      automatycznie używa GPU jeśli `torch.cuda.is_available()`
- [x] **Tabulate** w Dockerfile (potrzebne dla `pandas.to_markdown()` w `build_pdf.py`)

---

## 7. Sprawozdanie — struktura

PDF generowany z `report/report.md` przez WeasyPrint. Sekcje:

1. **Wstęp** — czym jest XAI, dlaczego ważne (shortcut learning, EU AI Act,
   debug błędów modelu). Cytat z Cholleta rozdz. 10.
2. **Co rozszerzamy względem książki** — Chollet pokazuje 1 metodę (Grad-CAM)
   na 1 modelu (Xception, Keras). My: 6 metod, 2 architektury (CNN + ViT),
   PyTorch + Captum, porównanie ilościowe.
3. **Przegląd metod** — sekcja per metoda: idea matematyczna w 5–10 zdaniach,
   wzór, pseudokod, ograniczenia.
4. **Implementacja** — krótkie omówienie architektury repo, jak uruchomić
   (`make up`), zrzuty z UI.
5. **Wyniki jakościowe** — galeria 8 obrazów × 6 metod, dla każdego komentarz:
   "tu Grad-CAM zgadza się z LIME ale IG dodaje X".
6. **Wyniki ilościowe** — tabela: średni czas inferencji per metoda, średnie
   IoU heatmap parami (macierz 6×6), dyskusja.
7. **Przypadki ciekawe** — np. snowy_wolf: czy model patrzy na zwierzę czy na
   śnieg (klasyczny przykład shortcut learning z papera LIME)?
8. **Dyskusja** — Grad-CAM vs IG: która "prawdziwsza"? Brak ground truth →
   nie da się rozstrzygnąć obiektywnie. Sanity checks Adebayo et al. 2018.
9. **Wnioski** — XAI to nie magia, każda metoda ma założenia. Praktyczne
   zalecenie: używaj ≥2 metod komplementarnych przed wdrożeniem modelu.
10. **Bibliografia** — Selvaraju 2017 (Grad-CAM), Sundararajan 2017 (IG),
    Smilkov 2017 (SmoothGrad), Zeiler 2014 (Occlusion), Ribeiro 2016 (LIME),
    Adebayo 2018 (Sanity checks), Chollet rozdz. 10.

**Długość faktyczna:** ~30 stron PDF (18 figur galerii + 2 macierze IoU).

---

## 8. Kryteria akceptacji (5.0)

- [ ] `make up` w czystym środowisku → apka działa pod `xai.local.pl`
- [ ] 6 metod XAI zaimplementowanych, każda zwraca poprawną heatmapę
- [ ] 2 architektury modeli (CNN + ViT) działają — z dyskusją różnic
- [ ] `make test` zielony (≥10 testów)
- [ ] `make report` generuje PDF ≥8 stron z figurami
- [ ] Sprawozdanie zawiera **porównanie ilościowe** (czas, IoU) — nie tylko
      jakościowe
- [ ] Sprawozdanie zawiera **dyskusję ograniczeń** (sanity checks, brak GT)
- [ ] README pozwala odtworzyć projekt komuś nieznającemu kontekstu

---

## 9. Ryzyka i mitygacje

| Ryzyko | Prawdopodobieństwo | Mitygacja |
|---|---|---|
| LIME bardzo wolny na CPU (>60 s/obraz) | wysokie | obniżyć `num_samples` z 1000 do 300, zaznaczyć w sprawozdaniu kompromis |
| ViT + Grad-CAM artefakty (token CLS) | średnie | użyć `reshape_transform` z dokumentacji pytorch-grad-cam, w sprawozdaniu pokazać przed/po |
| Obraz Dockera z torch ~3 GB | wysokie | obraz finalny ~6 GB z torch CUDA (kompromis: GPU passthrough zamiast CPU-only) |
| WeasyPrint na slim ma brakujące libki (cairo, pango) | średnie | apt-get install w Dockerfile (znana lista) |
| Streamlit + długi inference blokuje UI | średnie | `st.spinner` + `st.cache_resource` na model |

---

## 10. Linki referencyjne

- Chollet rozdz. 10: https://deeplearningwithpython.io/chapters/chapter10_interpreting-what-convnets-learn/
- Captum: https://captum.ai/
- pytorch-grad-cam: https://github.com/jacobgil/pytorch-grad-cam
- LIME: https://github.com/marcotcr/lime
- Selvaraju 2017 (Grad-CAM): https://arxiv.org/abs/1610.02391
- Sundararajan 2017 (Integrated Gradients): https://arxiv.org/abs/1703.01365
- Adebayo 2018 (Sanity checks): https://arxiv.org/abs/1810.03292
