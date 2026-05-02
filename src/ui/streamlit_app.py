"""xai-lab — Streamlit UI.

Upload obrazu → klasyfikacja na ImageNet → 6 heatmap XAI → tabela porównawcza.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image

from src.config import settings
from src.metrics.iou import pairwise_iou
from src.models.loader import ModelName, load_model, predict
from src.utils.imagenet import label_for, top_k as topk_predictions
from src.utils.preprocess import to_display_array, to_model_tensor
from src.utils.viz import overlay_heatmap
from src.xai.runner import run_all

st.set_page_config(page_title="xai-lab", layout="wide", page_icon="🔬")


@st.cache_resource(show_spinner="Ładuję model...")
def cached_load_model(name: ModelName, device: str):
    return load_model(name, device)


def render_predictions(probs: torch.Tensor) -> int:
    top5 = topk_predictions(probs, k=5)
    df = pd.DataFrame(
        [
            {"idx": idx, "klasa": label, "pewność": f"{p:.1%}"}
            for idx, label, p in top5
        ]
    )
    st.dataframe(df, hide_index=True, use_container_width=True)
    options = {f"{label} (idx {idx})": idx for idx, label, _ in top5}
    chosen_label = st.radio(
        "Klasa do wyjaśnienia:",
        list(options.keys()),
        horizontal=True,
        index=0,
    )
    return options[chosen_label]


def render_gallery(rgb: np.ndarray, results) -> None:
    cols = st.columns(3)
    for i, result in enumerate(results):
        with cols[i % 3]:
            overlay = overlay_heatmap(rgb, result.heatmap)
            caption = f"**{result.method}** — {result.elapsed_ms:.0f} ms"
            if result.extra and "error" in result.extra:
                caption = f"**{result.method}** — BŁĄD: {result.extra['error']}"
            st.image(overlay, caption=caption, use_container_width=True)


def render_iou_table(results) -> None:
    heatmaps = [r.heatmap for r in results]
    names = [r.method for r in results]
    matrix = pairwise_iou(heatmaps, top_pct=0.20)
    df = pd.DataFrame(matrix, index=names, columns=names).round(2)
    st.markdown(
        "**Macierz IoU (top 20% pikseli)** — 1.0 = identyczne, 0.0 = rozłączne. "
        "Im wyżej, tym bardziej metody zgadzają się *gdzie* model patrzy."
    )
    st.dataframe(df.style.background_gradient(cmap="Blues", axis=None), use_container_width=True)


def build_zip(rgb: np.ndarray, results, target_class: int, model_name: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("original.png", _png_bytes((rgb * 255).astype(np.uint8)))
        meta = {
            "model": model_name,
            "target_class": target_class,
            "target_label": label_for(target_class),
            "methods": [],
        }
        for r in results:
            overlay = overlay_heatmap(rgb, r.heatmap)
            slug = r.method.replace(" ", "_").replace("+", "p").replace("-", "_").lower()
            zf.writestr(f"{slug}_overlay.png", _png_bytes(overlay))
            zf.writestr(f"{slug}_heatmap.png", _png_bytes((r.heatmap * 255).astype(np.uint8)))
            meta["methods"].append(
                {
                    "name": r.method,
                    "elapsed_ms": r.elapsed_ms,
                    "extra": r.extra,
                }
            )
        zf.writestr("metadata.json", json.dumps(meta, indent=2, ensure_ascii=False))
    return buf.getvalue()


def _png_bytes(arr: np.ndarray) -> bytes:
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ───────────────────────────────────── UI ─────────────────────────────────────

st.title("🔬 xai-lab")
st.caption(
    "6 metod XAI obok siebie — Grad-CAM, Grad-CAM++, Integrated Gradients, "
    "SmoothGrad, Occlusion, LIME. Modele: ResNet50 (CNN) i ViT-B/16 (Transformer)."
)

with st.sidebar:
    st.header("Konfiguracja")
    model_labels = {"resnet50": "ResNet50 (CNN)", "vit_b_16": "ViT-B/16 (Transformer)"}
    model_name: ModelName = st.selectbox(
        "Model",
        options=["resnet50", "vit_b_16"],
        index=0 if settings.default_model == "resnet50" else 1,
        format_func=lambda x: model_labels[x],
    )
    cuda_ok = torch.cuda.is_available()
    device_options = ["cuda", "cpu"] if cuda_ok else ["cpu"]
    device_labels = {
        "cuda": f"GPU — {torch.cuda.get_device_name(0)}" if cuda_ok else "GPU (niedostępne)",
        "cpu": "CPU",
    }
    device = st.selectbox(
        "Urządzenie",
        options=device_options,
        index=0,
        format_func=lambda x: device_labels[x],
        help=(
            "GPU dostępne — domyślnie CUDA (~10× szybsze niż CPU)."
            if cuda_ok
            else "CUDA niedostępne (brak GPU passthrough lub torch CPU-only). "
                 "Sprawdź `docker info | grep nvidia`."
        ),
    )
    st.divider()
    st.markdown("**Domyślne hiperparametry XAI** (zmień w `.env`):")
    st.caption(
        f"- IG: {settings.ig_steps} kroków\n"
        f"- SmoothGrad: {settings.smoothgrad_samples} próbek, σ={settings.smoothgrad_sigma}\n"
        f"- Occlusion: patch {settings.occlusion_patch}, krok {settings.occlusion_stride}\n"
        f"- LIME: {settings.lime_samples} próbek, {settings.lime_segments} segmentów"
    )

samples_dir = Path("data/samples")
sample_files: list[Path] = []
if samples_dir.exists():
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        sample_files.extend(samples_dir.glob(ext))
    sample_files.sort()

source = st.radio(
    "Źródło obrazu:",
    options=["Wgranie własnego", "Próbka z data/samples/"] if sample_files else ["Wgranie własnego"],
    horizontal=True,
)

img: Image.Image | None = None
img_name: str = ""

if source == "Wgranie własnego":
    uploaded = st.file_uploader(
        "Wgraj obraz (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        img_name = uploaded.name
else:
    chosen = st.selectbox(
        "Wybierz obraz:",
        options=[f.name for f in sample_files],
    )
    chosen_path = samples_dir / chosen
    img = Image.open(chosen_path).convert("RGB")
    img_name = chosen

if img is None:
    st.info("Wgraj obraz aby zacząć.")
    st.stop()
rgb = to_display_array(img)
input_tensor = to_model_tensor(img, device=device)

col_left, col_right = st.columns([1, 2])
with col_left:
    st.image(rgb, caption=f"Wejście 224×224 — {img_name}", use_container_width=True)

with col_right:
    loaded = cached_load_model(model_name, device)
    probs = predict(loaded.model, input_tensor)
    st.subheader("Top-5 predykcji")
    target_class = render_predictions(probs)

st.divider()

if st.button("🔬 Wyjaśnij (uruchom 6 metod XAI)", type="primary"):
    spinner_msg = (
        "Generuję heatmapy — może potrwać 2–5 s na GPU..."
        if cuda_ok
        else "Generuję heatmapy — może potrwać 30–90 s na CPU..."
    )
    with st.spinner(spinner_msg):
        bundle = run_all(loaded, input_tensor, target_class)

    target_label = label_for(target_class)
    st.subheader(f"Heatmapy dla klasy: **{target_label}** (idx {target_class})")
    render_gallery(rgb, bundle.results)

    st.divider()
    st.subheader("Porównanie ilościowe")
    timing_df = pd.DataFrame(
        [{"metoda": r.method, "czas [ms]": f"{r.elapsed_ms:.0f}"} for r in bundle.results]
    )
    st.dataframe(timing_df, hide_index=True, use_container_width=True)
    render_iou_table(bundle.results)

    zip_bytes = build_zip(rgb, bundle.results, target_class, model_name)
    st.download_button(
        "⬇ Pobierz ZIP (heatmapy + metadane)",
        data=zip_bytes,
        file_name=f"xai_{model_name}_{target_class}.zip",
        mime="application/zip",
    )
