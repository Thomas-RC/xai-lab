"""Smoke test: 1 obraz, 1 model, 6 metod XAI — nie powinno crashnac."""

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.models.loader import load_model, predict
from src.utils.imagenet import top_k as topk
from src.utils.preprocess import to_display_array, to_model_tensor
from src.xai.runner import run_all


def _first_sample() -> Image.Image:
    samples = Path("data/samples")
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
        files.extend(samples.glob(ext))
    files.sort()
    if files:
        print(f"[smoke] uzywam: {files[0]}")
        return Image.open(files[0]).convert("RGB")
    print("[smoke] brak data/samples/*.jpg — generuje sztuczny obraz")
    arr = np.random.RandomState(0).randint(0, 255, (300, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def main() -> int:
    img = _first_sample()
    rgb = to_display_array(img)
    x = to_model_tensor(img, device="cpu")

    print("[smoke] laduje ResNet50...")
    loaded = load_model("resnet50", "cpu")

    print("[smoke] forward pass...")
    probs = predict(loaded.model, x)
    top = topk(probs, k=3)
    print(f"[smoke] top-3: {top}")
    target = top[0][0]

    print(f"[smoke] uruchamiam 6 metod XAI dla klasy {target}...")
    bundle = run_all(loaded, x, target)

    print("[smoke] wyniki:")
    for r in bundle.results:
        ok = "OK" if r.heatmap.max() > 0 and (not r.extra or "error" not in r.extra) else "FAIL"
        print(f"  - {r.method:25s}  {r.elapsed_ms:6.0f} ms  [{ok}]  shape={r.heatmap.shape}")

    failures = [r for r in bundle.results if r.extra and "error" in r.extra]
    if failures:
        print(f"\n[smoke] BLEDY: {len(failures)} metod nie powiodlo sie")
        for r in failures:
            print(f"  - {r.method}: {r.extra['error']}")
        return 1
    print(f"\n[smoke] OK — {len(bundle.results)}/6 metod zadzialalo, rgb.shape={rgb.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
