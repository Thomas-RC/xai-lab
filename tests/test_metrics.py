import numpy as np
import pytest

from src.metrics.iou import iou, pairwise_iou, top_k_mask
from src.metrics.timing import stopwatch


def test_iou_identical():
    a = np.random.rand(50, 50).astype(np.float32)
    assert iou(a, a) == pytest.approx(1.0)


def test_iou_disjoint():
    a = np.zeros((50, 50), dtype=np.float32)
    b = np.zeros((50, 50), dtype=np.float32)
    a[:10, :10] = 1.0
    b[40:, 40:] = 1.0
    assert iou(a, b, top_pct=0.04) == pytest.approx(0.0)


def test_iou_symmetric():
    rng = np.random.default_rng(0)
    a = rng.random((30, 30)).astype(np.float32)
    b = rng.random((30, 30)).astype(np.float32)
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_top_k_mask_count():
    arr = np.arange(100, dtype=np.float32).reshape(10, 10)
    mask = top_k_mask(arr, top_pct=0.20)
    assert mask.sum() == 20
    assert bool(mask[-1, -1]) is True


def test_pairwise_iou_shape_and_diag():
    rng = np.random.default_rng(0)
    heatmaps = [rng.random((20, 20)).astype(np.float32) for _ in range(4)]
    matrix = pairwise_iou(heatmaps)
    assert matrix.shape == (4, 4)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.allclose(matrix, matrix.T)


def test_iou_different_shapes_resamples():
    rng = np.random.default_rng(1)
    a = rng.random((100, 100)).astype(np.float32)
    b = rng.random((50, 50)).astype(np.float32)
    value = iou(a, b)
    assert 0.0 <= value <= 1.0


def test_stopwatch():
    import time

    with stopwatch() as t:
        time.sleep(0.01)
    assert t() >= 10.0
