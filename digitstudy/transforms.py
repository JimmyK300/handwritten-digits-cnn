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


# ------------------------------------------------------------------ domain-gap transforms


def _moments(img: np.ndarray):
    t = img.sum()
    if t <= 0:
        return 13.5, 13.5, 0.0
    yy, xx = np.mgrid[: img.shape[0], : img.shape[1]]
    cy, cx = (yy * img).sum() / t, (xx * img).sum() / t
    mu11 = ((xx - cx) * (yy - cy) * img).sum() / t
    mu02 = ((yy - cy) ** 2 * img).sum() / t
    return cy, cx, (mu11 / mu02 if mu02 > 0 else 0.0)


def _affine_about(img: np.ndarray, matrix: np.ndarray, center) -> np.ndarray:
    """Output pixel p samples input at matrix @ (p - center) + center (row, col order)."""
    from scipy.ndimage import affine_transform

    c = np.asarray(center, float)
    return np.clip(affine_transform(img, matrix, offset=c - matrix @ c, order=1, mode="constant"), 0, 1).astype(img.dtype)


def shear_to(images: np.ndarray, target_skew: np.ndarray | float = 0.0) -> np.ndarray:
    """Shear each image about its center of mass so its skew mu11/mu02 becomes `target_skew`.

    target_skew = 0 is classic deslanting (as in LeCun et al.'s deskewed MNIST)."""
    tgt = np.broadcast_to(np.asarray(target_skew, float), (len(images),))
    out = np.empty_like(images)
    for i, img in enumerate(images):
        cy, cx, skew = _moments(img)
        d = skew - tgt[i]  # x_in = x_out + d * (y - cy)
        out[i] = _affine_about(img, np.array([[1.0, 0.0], [d, 1.0]]), (cy, cx))
    return out


def hstretch(images: np.ndarray, factor: np.ndarray | float) -> np.ndarray:
    """Scale each image horizontally by `factor` about its center of mass."""
    f = np.broadcast_to(np.asarray(factor, float), (len(images),))
    out = np.empty_like(images)
    for i, img in enumerate(images):
        cy, cx, _ = _moments(img)
        out[i] = _affine_about(img, np.array([[1.0, 0.0], [0.0, 1.0 / f[i]]]), (cy, cx))
    return out


def peak_normalize(images: np.ndarray) -> np.ndarray:
    """Divide each image by its maximum so the brightest ink pixel is 1."""
    m = images.reshape(len(images), -1).max(1)
    return (images / np.maximum(m, 1e-6)[:, None, None]).astype(images.dtype)


def gaussian_blur(images: np.ndarray, sigma: float, renorm: bool = True) -> np.ndarray:
    """Soften edges with a Gaussian (sigma in px); renorm restores each image's peak."""
    from scipy.ndimage import gaussian_filter

    out = gaussian_filter(images, sigma=(0, sigma, sigma)).astype(images.dtype)
    if renorm:
        peak = images.reshape(len(images), -1).max(1) / np.maximum(out.reshape(len(out), -1).max(1), 1e-6)
        out = np.clip(out * peak[:, None, None], 0, 1).astype(images.dtype)
    return out


def binarize(images: np.ndarray, threshold: float = 0.5, value: float = 1.0) -> np.ndarray:
    return ((images > threshold) * value).astype(images.dtype)


def thin(images: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Zhang-Suen thinning of the binarized ink to one-pixel strokes (vectorized over the batch)."""
    b = np.pad(images > threshold, ((0, 0), (1, 1), (1, 1))).astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            P = b
            p2, p3, p4 = P[:, :-2, 1:-1], P[:, :-2, 2:], P[:, 1:-1, 2:]
            p5, p6, p7 = P[:, 2:, 2:], P[:, 2:, 1:-1], P[:, 2:, :-2]
            p8, p9 = P[:, 1:-1, :-2], P[:, :-2, :-2]
            nb = [p2, p3, p4, p5, p6, p7, p8, p9]
            B = sum(n.astype(np.int16) for n in nb)
            A = sum(((nb[k] == 0) & (nb[(k + 1) % 8] == 1)).astype(np.int16) for k in range(8))
            c = P[:, 1:-1, 1:-1] == 1
            if step == 0:
                m = c & (B >= 2) & (B <= 6) & (A == 1) & (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
            else:
                m = c & (B >= 2) & (B <= 6) & (A == 1) & (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
            if m.any():
                b[:, 1:-1, 1:-1][m] = 0
                changed = True
        if not changed:
            break
    return b[:, 1:-1, 1:-1].astype(images.dtype)


def shift_int(images: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Shift each image by whole pixels (dx right, dy down), zero-filled."""
    out = np.zeros_like(images)
    h, w = images.shape[1:]
    for i, img in enumerate(images):
        x, y = int(dx[i]), int(dy[i])
        ys, yd = (slice(0, h - y), slice(y, h)) if y >= 0 else (slice(-y, h), slice(0, h + y))
        xs, xd = (slice(0, w - x), slice(x, w)) if x >= 0 else (slice(-x, w), slice(0, w + x))
        out[i][yd, xd] = img[ys, xs]
    return out


def apply_preprocess(images: np.ndarray, spec: dict[str, Any] | None) -> np.ndarray:
    """Apply a deterministic preprocessing spec.

    spec keys (all optional, applied in this order):
      deslant: true                     -> shear each image to zero skew
      hstretch: float                   -> horizontal scale about the center of mass
      normalize: "mnist" | "center"     -> size/position normalization
      peak: true                        -> per-image max -> 1
      dilate_kernel, dilate_iterations  -> grayscale dilation
      erode_kernel, erode_iterations    -> grayscale erosion
      blur: float                       -> Gaussian blur sigma (peak restored)
      gain                              -> brightness gain (applied last)
    The original keys (normalize, dilate, erode, gain) keep their earlier behaviour.
    """
    if not spec:
        return images
    out = images
    if spec.get("deslant"):
        out = shear_to(out, 0.0)
    if spec.get("hstretch"):
        out = hstretch(out, float(spec["hstretch"]))
    if spec.get("normalize"):  # "mnist" (rescale + center) or "center" (center only)
        out = mnist_normalize(out, rescale=spec["normalize"] == "mnist")
    if spec.get("peak"):
        out = peak_normalize(out)
    if spec.get("dilate_kernel", 1) > 1 and spec.get("dilate_iterations", 0) > 0:
        out = dilate(out, int(spec["dilate_kernel"]), int(spec["dilate_iterations"]))
    if spec.get("erode_kernel", 1) > 1 and spec.get("erode_iterations", 0) > 0:
        out = erode(out, int(spec["erode_kernel"]), int(spec["erode_iterations"]))
    if spec.get("blur"):
        out = gaussian_blur(out, float(spec["blur"]))
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
