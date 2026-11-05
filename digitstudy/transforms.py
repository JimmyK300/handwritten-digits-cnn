"""Image preprocessing used in the study: grayscale morphology and brightness.

All functions take float images in [0, 1] shaped (N, H, W) or (H, W) and
return a new array with the same shape. Digits are white (1.0) on black (0.0),
as in MNIST, so dilation thickens strokes and erosion thins them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _window_reduce(images: np.ndarray, kernel: int, reduce, pad_value: float) -> np.ndarray:
    if kernel <= 1:
        return images.copy()
    squeeze = images.ndim == 2
    x = images[None] if squeeze else images
    before, after = (kernel - 1) // 2, kernel // 2
    padded = np.pad(
        x, ((0, 0), (before, after), (before, after)), mode="constant", constant_values=pad_value
    )
    windows = sliding_window_view(padded, (kernel, kernel), axis=(1, 2))
    out = reduce(windows, axis=(-2, -1)).astype(images.dtype, copy=False)
    return out[0] if squeeze else out


def dilate(images: np.ndarray, kernel: int = 3, iterations: int = 1) -> np.ndarray:
    """Grayscale dilation with a flat square `kernel` x `kernel` element."""
    out = images
    for _ in range(max(iterations, 0)):
        out = _window_reduce(out, kernel, np.max, 0.0)
    return out if iterations > 0 else images.copy()


def erode(images: np.ndarray, kernel: int = 3, iterations: int = 1) -> np.ndarray:
    """Grayscale erosion with a flat square element (background pads as 0)."""
    out = images
    for _ in range(max(iterations, 0)):
        out = _window_reduce(out, kernel, np.min, 0.0)
    return out if iterations > 0 else images.copy()


def brightness(images: np.ndarray, gain: float = 1.0) -> np.ndarray:
    """Scale intensities by `gain` and clip back to [0, 1]."""
    return np.clip(images * gain, 0.0, 1.0).astype(images.dtype, copy=False)


def _com_shift(img: np.ndarray) -> np.ndarray:
    """Shift by whole pixels so the intensity center of mass lands at the image center."""
    h, w = img.shape
    total = img.sum()
    if total <= 0:
        return img
    yy, xx = np.mgrid[:h, :w]
    dy = int(round((h - 1) / 2 - (yy * img).sum() / total))
    dx = int(round((w - 1) / 2 - (xx * img).sum() / total))
    out = np.zeros_like(img)
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def mnist_normalize(images: np.ndarray, box: int = 20, threshold: float = 0.05, rescale: bool = True) -> np.ndarray:
    """MNIST-style size and position normalization.

    Like the original MNIST pipeline: crop to the ink bounding box, scale (keeping
    aspect ratio, bilinear) so the longer side is `box` pixels, paste into a
    28x28 frame and shift so the center of mass is at the center.
    With rescale=False only the center-of-mass shift is applied.
    """
    from scipy.ndimage import zoom

    squeeze = images.ndim == 2
    x = images[None] if squeeze else images
    out = np.zeros_like(x)
    size = x.shape[-1]
    for i, img in enumerate(x):
        if not rescale:
            out[i] = _com_shift(img)
            continue
        rows = np.where(img.max(axis=1) > threshold)[0]
        cols = np.where(img.max(axis=0) > threshold)[0]
        if rows.size == 0:
            out[i] = img
            continue
        crop = img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
        scale = box / max(crop.shape)
        scaled = np.clip(zoom(crop, scale, order=1), 0.0, 1.0)[:box, :box]
        frame = np.zeros((size, size), dtype=img.dtype)
        top, left = (size - scaled.shape[0]) // 2, (size - scaled.shape[1]) // 2
        frame[top : top + scaled.shape[0], left : left + scaled.shape[1]] = scaled
        out[i] = _com_shift(frame)
    return out[0] if squeeze else out


def apply_preprocess(images: np.ndarray, spec: dict[str, Any] | None) -> np.ndarray:
    """Apply a deterministic preprocessing spec.

    spec keys (all optional, applied in this order):
      normalize: "mnist" | "center"     -> size/position normalization (first)
      dilate_kernel, dilate_iterations  -> grayscale dilation
      erode_kernel, erode_iterations    -> grayscale erosion
      gain                              -> brightness gain (applied last)
    """
    if not spec:
        return images
    out = images
    if spec.get("normalize"):  # "mnist" (rescale + center) or "center" (center only)
        out = mnist_normalize(out, rescale=spec["normalize"] == "mnist")
    if spec.get("dilate_kernel", 1) > 1 and spec.get("dilate_iterations", 0) > 0:
        out = dilate(out, int(spec["dilate_kernel"]), int(spec["dilate_iterations"]))
    if spec.get("erode_kernel", 1) > 1 and spec.get("erode_iterations", 0) > 0:
        out = erode(out, int(spec["erode_kernel"]), int(spec["erode_iterations"]))
    if spec.get("gain", 1.0) != 1.0:
        out = brightness(out, float(spec["gain"]))
    return out


def random_stroke_width(
    images: np.ndarray, rng: np.random.Generator, p_dilate: float = 0.33, p_erode: float = 0.33
) -> np.ndarray:
    """Train-time augmentation: per sample, dilate (3x3), erode (2x2) or leave as is."""
    out = images.copy()
    draw = rng.random(len(images))
    d_idx = draw < p_dilate
    e_idx = (draw >= p_dilate) & (draw < p_dilate + p_erode)
    if d_idx.any():
        out[d_idx] = dilate(images[d_idx], 3, 1)
    if e_idx.any():
        out[e_idx] = erode(images[e_idx], 2, 1)
    return out


def stroke_stats(images: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Cheap stroke statistics for comparing domains.

    ink_fraction: share of pixels above `threshold`.
    mean_ink: mean intensity of pixels above 0.05.
    thickness: ink pixels / skeleton-ish boundary estimate (ink / (ink - eroded ink)).
    """
    ink = images > threshold
    core = erode(images, 3, 1) > threshold
    n_ink = ink.sum(axis=(-2, -1)).astype(float)
    n_edge = np.maximum(n_ink - core.sum(axis=(-2, -1)), 1.0)
    lit = images[images > 0.05]
    return {
        "ink_fraction": float(ink.mean()),
        "mean_ink": float(lit.mean()) if lit.size else 0.0,
        "thickness_ratio": float(np.mean(n_ink / n_edge)),
    }
