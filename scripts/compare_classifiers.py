"""Porownanie 5 klasyfikatorow na zestawie testowym data/samples/.

Klasyfikatory:
- ResNet50 (CNN)        — ImageNet 1k, zamkniety slownik
- ViT-B/16 (Transformer) — ImageNet 1k, zamkniety slownik
- CLIP zero-shot         — ViT-B/32 (OpenAI 2021), open-vocab, NIE LLM
- SigLIP 2 zero-shot     — ViT-B/16 (Google luty 2025), open-vocab, NIE LLM
- Gemini 2.5 Flash       — Vision LLM via Vertex AI, otwarty slownik (free-text PL)

Output:
- `report/figures/classifier_comparison.csv`  — surowa tabela
- `report/figures/classifier_comparison.png`  — czytelna tabela do PDF

Uruchamiane przez `python -m scripts.compare_classifiers`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image

from src.models.clip_classifier import CLIPClassifier
from src.models.llm_classifier import LLMClassifier, VisionResponse
from src.models.loader import LoadedModel, load_model, predict
from src.models.siglip_classifier import SigLIPClassifier
from src.translation import translate_labels
from src.utils.imagenet import top_k as topk
from src.utils.preprocess import to_model_tensor

FIGS_DIR = Path("report/figures")
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def collect_samples() -> list[Path]:
    samples = Path("data/samples")
    files: list[Path] = []
    for ext in SAMPLE_EXTS:
        files.extend(samples.glob(ext))
    return sorted(files)


def torch_top1(loaded: LoadedModel, img: Image.Image) -> tuple[str, float]:
    x = to_model_tensor(img, device=DEVICE)
    probs = predict(loaded.model, x)
    idx, label, prob = topk(probs, k=1)[0]
    pl_label = translate_labels([label])[0]
    return f"{pl_label} ({label})", prob


def gemini_top1(clf: LLMClassifier, img: Image.Image) -> tuple[str, float, str]:
    resp: VisionResponse = clf.classify(img)
    top = resp.predictions[0]
    return top.label, top.confidence, resp.reasoning


def clip_top1(clf: CLIPClassifier, img: Image.Image) -> tuple[str, float]:
    resp = clf.classify(img, top_k=1)
    return resp.predictions[0].label, resp.predictions[0].confidence


def siglip_top1(clf: SigLIPClassifier, img: Image.Image) -> tuple[str, float]:
    resp = clf.classify(img, top_k=1)
    return resp.predictions[0].label, resp.predictions[0].confidence


def main() -> int:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    samples = collect_samples()
    if not samples:
        print("BLAD: brak obrazow w data/samples/", file=sys.stderr)
        return 1

    resnet = load_model("resnet50", DEVICE)
    vit = load_model("vit_b_16", DEVICE)
    gemini = load_model("gemini_vision", DEVICE)
    clip = CLIPClassifier(device=DEVICE)
    siglip = SigLIPClassifier(device=DEVICE)
    assert isinstance(resnet, LoadedModel)
    assert isinstance(vit, LoadedModel)
    assert isinstance(gemini, LLMClassifier)

    print(f"[compare] {len(samples)} obrazow x 5 modeli")
    rows: list[dict] = []
    for img_path in samples:
        print(f"\n[{img_path.name}]")
        img = Image.open(img_path).convert("RGB")

        try:
            r_label, r_prob = torch_top1(resnet, img)
            print(f"  ResNet50:      {r_label}  ({r_prob:.0%})")
        except Exception as exc:  # noqa: BLE001
            r_label, r_prob = f"BLAD: {exc}", 0.0

        try:
            v_label, v_prob = torch_top1(vit, img)
            print(f"  ViT-B/16:      {v_label}  ({v_prob:.0%})")
        except Exception as exc:  # noqa: BLE001
            v_label, v_prob = f"BLAD: {exc}", 0.0

        try:
            c_label, c_prob = clip_top1(clip, img)
            print(f"  CLIP:          {c_label}  ({c_prob:.0%})")
        except Exception as exc:  # noqa: BLE001
            c_label, c_prob = f"BLAD: {exc}", 0.0

        try:
            s_label, s_prob = siglip_top1(siglip, img)
            print(f"  SigLIP 2:      {s_label}  ({s_prob:.0%})")
        except Exception as exc:  # noqa: BLE001
            s_label, s_prob = f"BLAD: {exc}", 0.0

        try:
            g_label, g_prob, g_reasoning = gemini_top1(gemini, img)
            print(f"  Gemini Vision: {g_label}  ({g_prob:.0%})")
            print(f"  → {g_reasoning}")
        except Exception as exc:  # noqa: BLE001
            g_label, g_prob, g_reasoning = f"BLAD: {exc}", 0.0, ""

        rows.append(
            {
                "obraz": img_path.name,
                "ResNet50": f"{r_label}  ({r_prob:.0%})",
                "ViT-B/16": f"{v_label}  ({v_prob:.0%})",
                "CLIP": f"{c_label}  ({c_prob:.0%})",
                "SigLIP 2": f"{s_label}  ({s_prob:.0%})",
                "Gemini Vision": f"{g_label}  ({g_prob:.0%})",
                "Gemini reasoning": g_reasoning,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = FIGS_DIR / "classifier_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n→ {csv_path}")

    # PNG z tabela do PDF — bez kolumny reasoning (zbyt szeroka)
    df_png = df[["obraz", "ResNet50", "ViT-B/16", "CLIP", "SigLIP 2", "Gemini Vision"]]
    n = len(df_png)
    fig, ax = plt.subplots(figsize=(20, 0.55 * n + 1.2))
    ax.axis("off")
    table = ax.table(
        cellText=df_png.values,
        colLabels=df_png.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)
    for j in range(len(df_png.columns)):
        table[(0, j)].set_text_props(weight="bold")
        table[(0, j)].set_facecolor("#cfe2ff")
    ax.set_title("Porownanie 5 klasyfikatorow: ResNet50, ViT-B/16, CLIP, SigLIP 2, Gemini 2.5 Flash",
                 fontsize=11, weight="bold", pad=12)
    fig.tight_layout()
    png_path = FIGS_DIR / "classifier_comparison.png"
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {png_path}")

    print(f"\n[compare] OK — {len(rows)} wierszy")
    return 0


if __name__ == "__main__":
    sys.exit(main())
