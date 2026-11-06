"""Small shared statistics helpers."""

from __future__ import annotations

import numpy as np


def paired(a, b, scale: float = 100.0) -> tuple[float, float, float]:
    """Mean and std of (b - a) * scale, and the two-sided paired t-test p-value."""
    from scipy import stats

    d = scale * (np.asarray(b, float) - np.asarray(a, float))
    sd = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
    p = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 and sd > 0 else float("nan")
    return float(d.mean()), sd, p


def holm(pvals) -> list[float]:
    """Holm-Bonferroni adjusted p-values (same order as the input; NaN stays NaN)."""
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        out[i] = running
    return out.tolist()
