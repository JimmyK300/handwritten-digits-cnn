"""A small CNN written from scratch in NumPy: forward pass, backprop, Adam.

Architecture (default):
    conv 5x5 (1->8, pad 2) -> ReLU -> maxpool 2x2
    conv 3x3 (8->16, pad 1) -> ReLU -> maxpool 2x2
    flatten (16*7*7) -> dense 64 -> ReLU -> dense 10 -> softmax cross-entropy

Convolutions use im2col so each layer is one matrix multiply. Gradients are
checked against finite differences in tests/test_numpy_cnn.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------- layers


class Conv2D:
    def __init__(self, in_ch: int, out_ch: int, k: int, pad: int, rng: np.random.Generator, dtype=np.float32):
        scale = np.sqrt(2.0 / (in_ch * k * k))  # He init for ReLU
        self.params = {
            "W": (rng.standard_normal((out_ch, in_ch, k, k)) * scale).astype(dtype),
            "b": np.zeros(out_ch, dtype=dtype),
        }
        self.k, self.pad = k, pad
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        n, c, h, w = x.shape
        k, p = self.k, self.pad
        xp = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        ho, wo = h + 2 * p - k + 1, w + 2 * p - k + 1
        # (N, C, Ho, Wo, k, k) -> (N, Ho, Wo, C, k, k) -> rows
        win = sliding_window_view(xp, (k, k), axis=(2, 3))
        cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(n * ho * wo, c * k * k)
        wmat = self.params["W"].reshape(len(self.params["b"]), -1)
        out = cols @ wmat.T + self.params["b"]
        self.cache = (x.shape, cols, ho, wo)
        return out.reshape(n, ho, wo, -1).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        (n, c, h, w), cols, ho, wo = self.cache
        k, p = self.k, self.pad
        f = dout.shape[1]
        dflat = dout.transpose(0, 2, 3, 1).reshape(-1, f)
        wmat = self.params["W"].reshape(f, -1)
        self.grads["W"] = (dflat.T @ cols).reshape(self.params["W"].shape)
        self.grads["b"] = dflat.sum(axis=0)
        dcols = (dflat @ wmat).reshape(n, ho, wo, c, k, k)
        dxp = np.zeros((n, c, h + 2 * p, w + 2 * p), dtype=dout.dtype)
        for i in range(k):
            for j in range(k):
                dxp[:, :, i : i + ho, j : j + wo] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dxp[:, :, p : p + h, p : p + w] if p else dxp


class ReLU:
    params: dict = {}
    grads: dict = {}

    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class MaxPool2:
    params: dict = {}
    grads: dict = {}

    def forward(self, x):
        n, c, h, w = x.shape
        blocks = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
        idx = blocks.argmax(axis=-1)
        self.cache = (x.shape, idx)
        return np.take_along_axis(blocks, idx[..., None], axis=-1)[..., 0]

    def backward(self, dout):
        (n, c, h, w), idx = self.cache
        dblocks = np.zeros((n, c, h // 2, w // 2, 4), dtype=dout.dtype)
        np.put_along_axis(dblocks, idx[..., None], dout[..., None], axis=-1)
        return dblocks.reshape(n, c, h // 2, w // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h, w)


class Flatten:
    params: dict = {}
    grads: dict = {}

    def forward(self, x):
        self.shape = x.shape
        return x.reshape(len(x), -1)

    def backward(self, dout):
        return dout.reshape(self.shape)


class Dense:
    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, dtype=np.float32):
        self.params = {
            "W": (rng.standard_normal((d_in, d_out)) * np.sqrt(2.0 / d_in)).astype(dtype),
            "b": np.zeros(d_out, dtype=dtype),
        }
        self.grads: dict[str, np.ndarray] = {}

    def forward(self, x):
        self.x = x
        return x @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self.x.T @ dout
        self.grads["b"] = dout.sum(axis=0)
        return dout @ self.params["W"].T


def softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    """Mean cross-entropy and its gradient w.r.t. logits."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    n = len(labels)
    loss = -np.log(probs[np.arange(n), labels] + 1e-12).mean()
    grad = probs.copy()
    grad[np.arange(n), labels] -= 1.0
    return float(loss), grad / n


# ---------------------------------------------------------------- model


@dataclass
class NumpyCNNConfig:
    conv_channels: tuple[int, int] = (8, 16)
    kernel_sizes: tuple[int, int] = (5, 3)
    hidden_dim: int = 64
    num_classes: int = 10
    image_size: int = 28


class NumpyCNN:
    def __init__(self, config: NumpyCNNConfig | None = None, seed: int = 0, dtype=np.float32):
        cfg = config or NumpyCNNConfig()
        rng = np.random.default_rng(seed)
        c1, c2 = cfg.conv_channels
        k1, k2 = cfg.kernel_sizes
        side = cfg.image_size // 4
        self.layers = [
            Conv2D(1, c1, k1, k1 // 2, rng, dtype), ReLU(), MaxPool2(),
            Conv2D(c1, c2, k2, k2 // 2, rng, dtype), ReLU(), MaxPool2(),
            Flatten(),
            Dense(c2 * side * side, cfg.hidden_dim, rng, dtype), ReLU(),
            Dense(cfg.hidden_dim, cfg.num_classes, rng, dtype),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 3:
            x = x[:, None]
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dlogits: np.ndarray) -> None:
        d = dlogits
        for layer in reversed(self.layers):
            d = layer.backward(d)

    def parameters(self):
        for li, layer in enumerate(self.layers):
            for name in layer.params:
                yield (li, name), layer.params, layer.grads

    def num_params(self) -> int:
        return sum(p[name].size for (_, name), p, _ in self.parameters())

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        out = [self.forward(x[i : i + batch_size]).argmax(axis=1) for i in range(0, len(x), batch_size)]
        return np.concatenate(out)


@dataclass
class Adam:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    t: int = 0
    m: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)

    def step(self, model: NumpyCNN) -> None:
        self.t += 1
        for key, params, grads in model.parameters():
            name = key[1]
            g = grads[name]
            if key not in self.m:
                self.m[key] = np.zeros_like(g)
                self.v[key] = np.zeros_like(g)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g * g
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[name] -= (self.lr * m_hat / (np.sqrt(v_hat) + self.eps)).astype(params[name].dtype)


def train_numpy_cnn(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 0,
    augment=None,
    config: NumpyCNNConfig | None = None,
    log=print,
) -> tuple[NumpyCNN, list[float]]:
    """Mini-batch training. `augment(batch, rng)` optionally transforms each batch."""
    model = NumpyCNN(config, seed=seed)
    opt = Adam(lr=lr)
    rng = np.random.default_rng(seed + 1)
    history: list[float] = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total, count = 0.0, 0
        for start in range(0, len(x), batch_size):
            idx = order[start : start + batch_size]
            xb = x[idx]
            if augment is not None:
                xb = augment(xb, rng)
            logits = model.forward(xb)
            loss, dlogits = softmax_cross_entropy(logits, y[idx])
            model.backward(dlogits.astype(logits.dtype))
            opt.step(model)
            total += loss * len(idx)
            count += len(idx)
        history.append(total / count)
        if log:
            log(f"  numpy-cnn epoch {epoch + 1}/{epochs} loss={history[-1]:.4f}")
    return model, history
