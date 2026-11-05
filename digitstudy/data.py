"""Load the team's stylus dataset (per writer) and MNIST from IDX files.

Images are returned as float32 arrays in [0, 1] with shape (N, 28, 28);
labels as int64 arrays with shape (N,).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
STYLUS_ROOT = REPO_ROOT / "mydata"
MNIST_ROOT = REPO_ROOT / "mnist"


def read_idx(path: str | Path) -> np.ndarray:
    """Read an IDX (ubyte) file into a uint8 array with its declared shape."""
    raw = Path(path).read_bytes()
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0:
        raise ValueError(f"Not an IDX file: {path}")
    if raw[2] != 0x08:
        raise ValueError(f"Only uint8 IDX files are supported: {path}")
    ndim = raw[3]
    dims = [int.from_bytes(raw[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim)]
    data = np.frombuffer(raw, dtype=np.uint8, offset=4 + 4 * ndim)
    expected = int(np.prod(dims))
    if data.size != expected:
        raise ValueError(f"{path}: header says {expected} values, file has {data.size}")
    return data.reshape(dims)


@dataclass
class WriterData:
    name: str
    images: np.ndarray  # (N, 28, 28) float32 in [0, 1]
    labels: np.ndarray  # (N,) int64
    # True for samples the team originally put in their t10k (test) file.
    team_test_mask: np.ndarray


def _to_float(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def list_writers(root: Path = STYLUS_ROOT) -> list[str]:
    return sorted(
        p.name for p in root.iterdir() if p.is_dir() and (p / "train-images-idx3-ubyte").exists()
    )


def load_writer(name: str, root: Path = STYLUS_ROOT) -> WriterData:
    folder = root / name
    tr_x = read_idx(folder / "train-images-idx3-ubyte")
    tr_y = read_idx(folder / "train-labels-idx1-ubyte")
    te_x = read_idx(folder / "t10k-images-idx3-ubyte")
    te_y = read_idx(folder / "t10k-labels-idx1-ubyte")
    images = _to_float(np.concatenate([tr_x, te_x]))
    labels = np.concatenate([tr_y, te_y]).astype(np.int64)
    mask = np.concatenate([np.zeros(len(tr_y), bool), np.ones(len(te_y), bool)])
    return WriterData(name, images, labels, mask)


def load_stylus(root: Path = STYLUS_ROOT) -> dict[str, WriterData]:
    return {name: load_writer(name, root) for name in list_writers(root)}


def load_mnist(split: str = "train", root: Path = MNIST_ROOT) -> tuple[np.ndarray, np.ndarray]:
    prefix = "train" if split == "train" else "t10k"
    x = read_idx(root / f"{prefix}-images-idx3-ubyte")
    y = read_idx(root / f"{prefix}-labels-idx1-ubyte")
    return _to_float(x), y.astype(np.int64)


def concat(writers: list[WriterData]) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate([w.images for w in writers]),
        np.concatenate([w.labels for w in writers]),
    )
