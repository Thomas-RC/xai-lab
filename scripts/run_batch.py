"""Batch runner do generacji figur i metryk dla sprawozdania.

Iteruje po `data/samples/*.{jpg,png}`, dla kazdego obrazu uruchamia 6 metod XAI
na obu modelach (ResNet50 + ViT), zapisuje:
- `report/figures/<image>_<model>.png`  — galeria 2x3 + oryginal
- `report/figures/iou_<model>_mean.png` — heatmapa srednich IoU
- `report/figures/timing_summary.csv`   — czasy inferencji
- `report/figures/predictions.csv`      — top-1 predykcje per (obraz, model)

Uruchamiane przez `make report-figures` lub `make report`.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image

from src.metrics.iou import pairwise_iou
from src.models.loader import ModelName, load_model, predict
from src.utils.imagenet import top_k as topk
from src.utils.preprocess import to_display_array, to_model_tensor
from src.utils.viz import gallery_figure
from src.xai.runner import run_all

MODELS: list[ModelName] = ["resnet50", "vit_b_16"]
FIGS_DIR = Path("report/figures")
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


SAMPLE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp")


def collect_samples() -> list[Path]:
    samples = Path("data/samples")
    files: list[Path] = []
    for ext in SAMPLE_EXTS:
        files.extend(samples.glob(ext))
    return sorted(files)


def process_one(image_path: Path, model_name: ModelName) -> dict:
    img = Image.open(image_path).convert("RGB")
    rgb = to_display_array(img)
    x = to_model_tensor(img, device=DEVICE)

    loaded = load_model(model_name, DEVICE)
    probs = predict(loaded.model, x)
    top1_idx, top1_label, top1_prob = topk(probs, k=1)[0]

    bundle = run_all(loaded, x, top1_idx)

    title = f"{image_path.name} → {top1_label} ({top1_prob:.0%}) [{model_name}]"
    fig = gallery_figure(rgb, bundle.results, title=title)
    out_path = FIGS_DIR / f"{image_path.stem}_{model_name}.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out_path}")

    return {
        "image": image_path.name,
        "model": model_name,
        "target_class": top1_idx,
        "target_label": top1_label,
        "target_prob": top1_prob,
        "results": bundle.results,
    }


def aggregate_iou(rows: list[dict], model_name: ModelName) -> pd.DataFrame:
    """Srednie IoU parami metod, usrednione po obrazach dla danego modelu."""
    rows_for_model = [r for r in rows if r["model"] == model_name]
    if not rows_for_model:
        return pd.DataFrame()
    method_names = [r.method for r in rows_for_model[0]["results"]]
    n = len(method_names)
    accumulator = np.zeros((n, n), dtype=np.float64)
    for row in rows_for_model:
        heatmaps = [r.heatmap for r in row["results"]]
        accumulator += pairwise_iou(heatmaps, top_pct=0.20)
    matrix = accumulator / len(rows_for_model)
    return pd.DataFrame(matrix, index=method_names, columns=method_names)


def save_iou_heatmap(df: pd.DataFrame, model_name: ModelName) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(df.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(df.columns)), df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(df.index)), df.index)
    for i in range(len(df.index)):
        for j in range(len(df.columns)):
            ax.text(j, i, f"{df.values[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title(f"Średnie IoU (top 20%) — {model_name}")
    fig.colorbar(im)
    fig.tight_layout()
    out = FIGS_DIR / f"iou_{model_name}_mean.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  → {out}")


def main() -> int:
    FIGS_DIR.mkdir(parents=True, exist_ok=True)
    samples = collect_samples()
    if not samples:
        print("BLAD: brak obrazow w data/samples/. Dodaj kilka .jpg/.png.")
        return 1

    print(f"[batch] {len(samples)} obrazow x {len(MODELS)} modele = "
          f"{len(samples) * len(MODELS)} przebiegow")

    rows: list[dict] = []
    for img_path in samples:
        for model_name in MODELS:
            print(f"\n[{img_path.name} | {model_name}]")
            try:
                rows.append(process_one(img_path, model_name))
            except Exception as exc:  # noqa: BLE001
                print(f"  ! BLAD: {exc}")

    # Timing CSV
    timing_records: list[dict] = []
    for row in rows:
        for r in row["results"]:
            timing_records.append(
                {
                    "image": row["image"],
                    "model": row["model"],
                    "method": r.method,
                    "elapsed_ms": r.elapsed_ms,
                }
            )
    pd.DataFrame(timing_records).to_csv(FIGS_DIR / "timing_summary.csv", index=False)
    print(f"\n→ {FIGS_DIR / 'timing_summary.csv'}")

    # Predictions CSV
    pred_records = [
        {
            "image": row["image"],
            "model": row["model"],
            "target_class": row["target_class"],
            "target_label": row["target_label"],
            "target_prob": row["target_prob"],
        }
        for row in rows
    ]
    pd.DataFrame(pred_records).to_csv(FIGS_DIR / "predictions.csv", index=False)
    print(f"→ {FIGS_DIR / 'predictions.csv'}")

    # IoU per model
    for model_name in MODELS:
        df = aggregate_iou(rows, model_name)
        if not df.empty:
            df.to_csv(FIGS_DIR / f"iou_{model_name}.csv")
            save_iou_heatmap(df, model_name)

    # Manifest dla build_pdf.py
    manifest = {
        "samples": [s.name for s in samples],
        "models": MODELS,
        "n_runs": len(rows),
    }
    (FIGS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n[batch] OK — {len(rows)} przebiegow zakonczonych")
    return 0


if __name__ == "__main__":
    sys.exit(main())
