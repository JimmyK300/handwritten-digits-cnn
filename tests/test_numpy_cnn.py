"""Finite-difference gradient checks for the from-scratch NumPy CNN."""

import numpy as np

from digitstudy.numpy_cnn import (
    Conv2D, Dense, MaxPool2, NumpyCNN, NumpyCNNConfig, softmax_cross_entropy, train_numpy_cnn,
)


def _num_grad(f, arr, eps=1e-6):
    g = np.zeros_like(arr)
    it = np.nditer(arr, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = arr[i]
        arr[i] = old + eps
        fp = f()
        arr[i] = old - eps
        fm = f()
        arr[i] = old
        g[i] = (fp - fm) / (2 * eps)
    return g


def _rel_err(a, b):
    return np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b)))


def test_conv_backward_matches_numeric():
    rng = np.random.default_rng(0)
    conv = Conv2D(2, 3, 3, 1, rng, dtype=np.float64)
    x = rng.standard_normal((2, 2, 5, 5))
    w = rng.standard_normal((2, 3, 5, 5))
    loss = lambda: float((conv.forward(x) * w).sum())  # noqa: E731
    loss()
    dx = conv.backward(w)
    assert _rel_err(dx, _num_grad(loss, x)) < 1e-6
    assert _rel_err(conv.grads["W"], _num_grad(loss, conv.params["W"])) < 1e-6
    assert _rel_err(conv.grads["b"], _num_grad(loss, conv.params["b"])) < 1e-6


def test_maxpool_and_dense_backward():
    rng = np.random.default_rng(1)
    pool = MaxPool2()
    x = rng.standard_normal((2, 3, 4, 4))
    w = rng.standard_normal((2, 3, 2, 2))
    loss = lambda: float((pool.forward(x) * w).sum())  # noqa: E731
    loss()
    assert _rel_err(pool.backward(w), _num_grad(loss, x)) < 1e-6

    dense = Dense(6, 4, rng, dtype=np.float64)
    xd = rng.standard_normal((3, 6))
    wd = rng.standard_normal((3, 4))
    loss = lambda: float((dense.forward(xd) * wd).sum())  # noqa: E731
    loss()
    assert _rel_err(dense.backward(wd), _num_grad(loss, xd)) < 1e-6
    assert _rel_err(dense.grads["W"], _num_grad(loss, dense.params["W"])) < 1e-6


def test_full_network_gradient():
    cfg = NumpyCNNConfig(conv_channels=(2, 3), kernel_sizes=(3, 3), hidden_dim=5, image_size=8)
    model = NumpyCNN(cfg, seed=3, dtype=np.float64)
    rng = np.random.default_rng(4)
    x = rng.random((3, 8, 8))
    y = np.array([0, 3, 7])

    def loss():
        return softmax_cross_entropy(model.forward(x), y)[0]

    _, dlogits = softmax_cross_entropy(model.forward(x), y)
    model.backward(dlogits)
    for (li, name), params, grads in model.parameters():
        analytic = grads[name].copy()
        numeric = _num_grad(loss, params[name])
        assert _rel_err(analytic, numeric) < 1e-5, (li, name)


def test_training_learns_a_separable_toy_problem():
    rng = np.random.default_rng(0)
    x = np.zeros((200, 28, 28), dtype=np.float32)
    y = rng.integers(0, 2, 200)
    x[y == 0, 5:10, 5:23] = 1.0  # horizontal bar
    x[y == 1, 5:23, 12:16] = 1.0  # vertical bar
    model, hist = train_numpy_cnn(x, y, epochs=3, batch_size=32, seed=0, log=None)
    assert hist[-1] < hist[0]
    assert (model.predict(x) == y).mean() > 0.95
