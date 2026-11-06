import numpy as np
import pytest

from digitstudy import gapstats as G
from digitstudy import transforms as T
from digitstudy.stats import holm


def _bar(width: int, slant: float = 0.0) -> np.ndarray:
    img = np.zeros((1, 28, 28), np.float32)
    for y in range(4, 24):
        x0 = int(round(12 + slant * (y - 14)))
        img[0, y, x0 : x0 + width] = 1.0
    return img


def test_stroke_width_measure():
    for w in (1, 3, 5):
        assert G.measure(_bar(w))["stroke_width"][0] == pytest.approx(w, abs=0.6)


def test_thin_gives_one_pixel_strokes():
    thick = _bar(5)
    assert G.measure(T.thin(thick))["stroke_width"][0] == pytest.approx(1.0, abs=0.3)


def test_shear_to_removes_slant():
    img = _bar(3, slant=0.4)
    before = abs(G.measure(img)["slant_deg"][0])
    after = abs(G.measure(T.shear_to(img, 0.0))["slant_deg"][0])
    assert before > 10 and after < 3


def test_hstretch_widens():
    img = np.zeros((1, 28, 28), np.float32)
    img[0, 6:22, 10:18] = 1
    w0 = G.measure(img)["bbox_w"][0]
    assert G.measure(T.hstretch(img, 1.5))["bbox_w"][0] > w0 * 1.3


def test_holm_matches_definition():
    p = [0.01, 0.04, 0.03, 0.2]
    # sorted: 0.01*4=0.04, 0.03*3=0.09, 0.04*2=0.08 -> monotone 0.09, 0.2*1 -> 0.2
    assert np.allclose(holm(p), [0.04, 0.09, 0.09, 0.2])
    assert np.isnan(holm([np.nan, 0.5])[0])


def test_augment_batch_shapes_and_range():
    torch = pytest.importorskip("torch")
    from digitstudy.domain_gap import augment_batch

    x = torch.from_numpy(np.concatenate([_bar(2)] * 6)).unsqueeze(1)
    g = torch.Generator().manual_seed(0)
    for a in ["rand_dilate", "rand_erode", "noise", "blur", "contrast", "hard", "slant", "affine", "elastic"]:
        y = augment_batch(x.clone(), [a], g)
        assert y.shape == x.shape
        assert float(y.min()) >= 0 and float(y.max()) <= 1.0001
