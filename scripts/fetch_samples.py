"""Pobiera przykladowe obrazy do `data/samples/` z publicznych zasobow.

Uzywa stabilnych URL z repo `pytorch-grad-cam` (autor jacobgil) — te same
obrazy sa uzywane w tutorialach biblioteki XAI, dzieki czemu wyniki da sie
porownac z dokumentacja oryginalu.

Uruchamiac: `python -m scripts.fetch_samples`
"""

import sys
import urllib.request
from pathlib import Path

# Stabilne, publiczne obrazy testowe z popularnych projektow XAI.
# Mozesz je zastapic / dodac wlasne JPG do data/samples/.
SAMPLES = [
    (
        "both.png",
        "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/both.png",
    ),
    (
        "dogs.png",
        "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/dogs.png",
    ),
    (
        "puppies.jpg",
        "https://raw.githubusercontent.com/jacobgil/pytorch-grad-cam/master/examples/puppies.jpg",
    ),
]


def main() -> int:
    out_dir = Path("data/samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, url in SAMPLES:
        target = out_dir / name
        if target.exists():
            print(f"[skip] {target} juz istnieje")
            continue
        print(f"[fetch] {url} -> {target}")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! BLAD: {exc}")
            return 1
    print(f"\n[fetch] gotowe. Mozesz dodac wiecej obrazow recznie do {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
