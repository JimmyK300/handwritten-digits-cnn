import numpy as np

from digitstudy.transforms import apply_preprocess, brightness, dilate, erode


def _dot():
    img = np.zeros((7, 7), dtype=np.float32)
    img[3, 3] = 1.0
    return img


def test_dilate_3x3_grows_a_dot_to_a_square():
    out = dilate(_dot(), 3, 1)
    assert out.sum() == 9 and out[2:5, 2:5].min() == 1.0


def test_dilate_iterations_compose():
    np.testing.assert_array_equal(dilate(_dot(), 3, 2), dilate(dilate(_dot(), 3, 1), 3, 1))
    assert dilate(_dot(), 3, 2).sum() == 25


def test_dilate_2x2_and_batch_shape():
    batch = np.stack([_dot(), _dot()])
    out = dilate(batch, 2, 1)
    assert out.shape == batch.shape and out[0].sum() == 4


def test_erode_inverts_dilate_on_a_square():
    sq = np.zeros((9, 9), dtype=np.float32)
    sq[2:7, 2:7] = 1.0
    np.testing.assert_array_equal(erode(dilate(sq, 3, 1), 3, 1), sq)


def test_brightness_clips():
    img = np.array([[0.2, 0.8]], dtype=np.float32)
    np.testing.assert_allclose(brightness(img, 2.0), [[0.4, 1.0]])


def test_apply_preprocess_empty_is_identity():
    img = np.random.default_rng(0).random((2, 5, 5)).astype(np.float32)
    assert apply_preprocess(img, {}) is img
    np.testing.assert_array_equal(apply_preprocess(img, {"dilate_kernel": 3, "dilate_iterations": 1}), dilate(img, 3, 1))
