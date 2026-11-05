"""PyTorch CNN and an in-memory training loop.

The architecture is the config-driven CNN from JimmyK300/autoresearch-cnn-rocm
(itself built on the team's myAPI.model.CNN). The default config in
configs/study.json matches the team's sampleModel/config.json.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        conv_channels: list[int] = (32, 32),
        kernel_sizes: list[int] = (5, 3),
        paddings: list[int] = (2, 1),
        hidden_dims: list[int] = (128, 32),
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        if not (len(conv_channels) == len(kernel_sizes) == len(paddings)):
            raise ValueError("conv_channels, kernel_sizes, and paddings must align.")
        layers: list[nn.Module] = []
        current = in_channels
        for out_ch, k, p in zip(conv_channels, kernel_sizes, paddings):
            layers += [nn.Conv2d(current, out_ch, kernel_size=k, padding=p), nn.ReLU(), nn.MaxPool2d(2)]
            current = out_ch
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat = int(self.features(torch.zeros(1, in_channels, 28, 28)).numel())
        head: list[nn.Module] = [nn.Flatten()]
        for h in hidden_dims:
            head += [nn.Linear(flat, h), nn.ReLU()]
            flat = h
        head.append(nn.Linear(flat, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_torch(
    x: np.ndarray,
    y: np.ndarray,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    seed: int,
    augment: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None,
) -> tuple[CNN, dict[str, Any]]:
    """Train on in-memory arrays. `augment` is applied to the whole set once per epoch."""
    set_seed(seed)
    model = CNN(**model_cfg)
    opt = torch.optim.Adam(model.parameters(), lr=float(train_cfg["learning_rate"]))
    loss_fn = nn.CrossEntropyLoss()
    bs = int(train_cfg["batch_size"])
    rng = np.random.default_rng(seed + 1)
    y_t = torch.from_numpy(y)
    losses: list[float] = []
    started = time.time()
    for _ in range(int(train_cfg["epochs"])):
        xe = augment(x, rng) if augment is not None else x
        x_t = torch.from_numpy(np.ascontiguousarray(xe, dtype=np.float32)).unsqueeze(1)
        order = torch.from_numpy(rng.permutation(len(x)))
        model.train()
        total = 0.0
        for start in range(0, len(x), bs):
            idx = order[start : start + bs]
            opt.zero_grad()
            loss = loss_fn(model(x_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        losses.append(total / len(x))
    info = {
        "train_loss_history": losses,
        "train_seconds": time.time() - started,
        "num_params": sum(p.numel() for p in model.parameters()),
    }
    return model, info


@torch.no_grad()
def predict_torch(model: CNN, x: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(np.ascontiguousarray(x[start : start + batch_size], dtype=np.float32)).unsqueeze(1)
        out.append(model(xb).argmax(dim=1).numpy())
    return np.concatenate(out)
