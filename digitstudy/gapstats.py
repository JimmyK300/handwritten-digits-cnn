"""Per-image measurements that describe the MNIST vs stylus domain gap.

Every function takes float images in [0, 1], shape (N, 28, 28), white ink on black.
Ink is binarized at INK = 0.25 for the shape measures, the same threshold for
both domains.

  stroke_width   2 * mean distance-transform value on the stroke ridge - 1
                 (1 for a one-pixel line, 3 for a three-pixel line)
  mean_ink       mean intensity of pixels > 0.02
  midgray_frac   share of pixels > 0.02 that lie in (0.1, 0.9): anti-aliasing / soft edges
  edge_softness  mean of 1 - |2v - 1| over pixels > 0.02 (0 = hard binary edge, 1 = all 0.5)
  bbox_h, bbox_w, bbox_max, aspect (w / h)
  com_offset     distance (px) from the intensity center of mass to the image center
  com_dx, com_dy
  slant_deg      atan(mu11 / mu02) of the ink: the shear that deslanting removes
  coverage       share of the 784 pixels > 0.02
  ink_coverage   share of pixels > INK
  n_components   8-connected ink components; stray = components of <= 3 px
  holes          enclosed background regions (closed loops)
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

INK = 0.25
_EIGHT = np.ones((3, 3), bool)


def _ridge_width(b: np.ndarray) -> float:
    if not b.any():
        return 0.0
    dt = ndi.distance_transform_edt(b)
    ridge = b & (dt >= ndi.maximum_filter(dt, size=3))
    return float(2 * dt[ridge].mean() - 1)


def measure(images: np.ndarray) -> dict[str, np.ndarray]:
    n = len(images)
    out = {k: np.zeros(n) for k in ("stroke_width", "mean_ink", "midgray_frac", "edge_softness", "bbox_h", "bbox_w",
                                     "bbox_max", "aspect", "com_offset", "com_dx", "com_dy", "slant_deg", "coverage",
                                     "ink_coverage", "n_components", "stray_components", "holes")}
    yy, xx = np.mgrid[:28, :28]
    for i, img in enumerate(images):
        lit = img > 0.02
        v = img[lit]
        b = img > INK
        out["coverage"][i] = lit.mean()
        out["ink_coverage"][i] = b.mean()
        out["mean_ink"][i] = v.mean() if v.size else 0
        out["midgray_frac"][i] = ((v > 0.1) & (v < 0.9)).mean() if v.size else 0
        out["edge_softness"][i] = (1 - np.abs(2 * v - 1)).mean() if v.size else 0
        out["stroke_width"][i] = _ridge_width(b)
        rows, cols = np.where(b.any(1))[0], np.where(b.any(0))[0]
        if rows.size:
            h, w = rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1
            out["bbox_h"][i], out["bbox_w"][i], out["bbox_max"][i], out["aspect"][i] = h, w, max(h, w), w / h
        t = img.sum()
        if t > 0:
            cy, cx = (yy * img).sum() / t, (xx * img).sum() / t
            out["com_dx"][i], out["com_dy"][i] = cx - 13.5, cy - 13.5
            out["com_offset"][i] = np.hypot(cx - 13.5, cy - 13.5)
            mu11 = ((xx - cx) * (yy - cy) * img).sum() / t
            mu02 = ((yy - cy) ** 2 * img).sum() / t
            out["slant_deg"][i] = np.degrees(np.arctan2(mu11, mu02))
        lab, k = ndi.label(b, structure=_EIGHT)
        out["n_components"][i] = k
        if k:
            sizes = np.bincount(lab.ravel())[1:]
            out["stray_components"][i] = int((sizes <= 3).sum())
        bg, kb = ndi.label(~b)  # 4-connected background
        if kb:
            border = set(np.unique(np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]])))
            out["holes"][i] = sum(1 for j in range(1, kb + 1) if j not in border)
    return out


FACTORS = [
    # (key, label, unit, group)
    ("stroke_width", "Stroke width (ridge)", "px", "Stroke width"),
    ("mean_ink", "Mean ink intensity", "0-1", "Intensity / anti-aliasing"),
    ("midgray_frac", "Mid-gray share of ink pixels", "share", "Intensity / anti-aliasing"),
    ("edge_softness", "Edge softness", "0-1", "Intensity / anti-aliasing"),
    ("bbox_h", "Bounding-box height", "px", "Size"),
    ("bbox_w", "Bounding-box width", "px", "Size"),
    ("bbox_max", "Bounding-box longer side (MNIST convention: 20)", "px", "Size"),
    ("aspect", "Aspect ratio w / h", "ratio", "Size"),
    ("com_offset", "Center-of-mass offset from center", "px", "Position"),
    ("slant_deg", "Slant (second-moment angle)", "deg", "Slant"),
    ("coverage", "Ink coverage (> 0.02)", "share", "Coverage"),
    ("ink_coverage", "Ink coverage (> 0.25)", "share", "Coverage"),
    ("n_components", "Connected ink components", "count", "Device artifacts"),
    ("stray_components", "Stray specks (components <= 3 px)", "count", "Device artifacts"),
    ("holes", "Closed loops (holes)", "count", "Shape"),
]
