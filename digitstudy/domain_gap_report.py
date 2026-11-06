"""Tables and figures for the MNIST <-> stylus domain-gap study -> results/domain_gap/DOMAIN_GAP.md."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from digitstudy.domain_gap import _prereg, _raw, load_config, load_runs
from digitstudy.gapstats import FACTORS
from digitstudy.stats import holm, paired

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]


def _scores(runs, direction, phase="final", field="test"):
    """recipe -> {seed: score}"""
    out: dict[str, dict[int, float]] = defaultdict(dict)
    for r in runs:
        if r["phase"] == phase and r["direction"] == direction:
            out[r["recipe"]][r["seed"]] = r[field]["score"] if field == "test" else r[field]["acc"]
    return out


def _cm(runs, direction, recipe):
    ms = [np.array(r["test"]["confusion"]) for r in runs
          if r["phase"] == "final" and r["direction"] == direction and r["recipe"] == recipe]
    return ms


def class_metrics(cm: np.ndarray) -> dict[str, float]:
    cm = np.asarray(cm, float)
    return {"recall_7": cm[7, 7] / cm[7].sum(), "precision_8": cm[8, 8] / max(cm[:, 8].sum(), 1),
            "pred_share_8": cm[:, 8].sum() / cm.sum(), "rate_9_as_4": cm[9, 4] / cm[9].sum(),
            "recall_9": cm[9, 9] / cm[9].sum(), "precision_4": cm[4, 4] / max(cm[:, 4].sum(), 1)}


def comparisons(runs, cfg) -> list[dict[str, Any]]:
    """Every recipe vs baseline on the test set, both directions, with Holm over the exploratory ones."""
    pre = _prereg(cfg)
    rows = []
    for d in ("s2m", "m2s"):
        sc = _scores(runs, d)
        val = _scores(runs, d, field="val")
        sel = _scores(runs, d, phase="select", field="val")
        base = sc.get("baseline", {})
        for rec, seeds in sc.items():
            common = sorted(set(seeds) & set(base))
            vals = [seeds[s] for s in sorted(seeds)]
            row = {"direction": d, "recipe": rec, "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                   "val_mean": float(np.mean(list(val[rec].values()))) if rec in val else None,
                   "select_val": float(np.mean(list(sel[rec].values()))) if rec in sel else None,
                   "headline": rec == pre.get(f"headline_{d}"), "n": len(vals)}
            if rec != "baseline" and len(common) > 1:
                dm, ds, p = paired([base[s] for s in common], [seeds[s] for s in common])
                row.update({"delta": dm, "delta_std": ds, "p": p,
                            "improved": int(sum(seeds[s] > base[s] for s in common)), "n_paired": len(common)})
            rows.append(row)
    expl = [r for r in rows if "p" in r and not r["headline"]]
    for r, a in zip(expl, holm([r["p"] for r in expl])):
        r["p_holm"] = a
    return rows


def _fmt(m, s=None, scale=100, d=1):
    if m is None:
        return "—"
    return f"{scale * m:.{d}f}" + (f" ± {scale * s:.{d}f}" if s is not None else "")


def _p(p):
    if p is None or np.isnan(p):
        return "—"
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def markdown(cfg, runs, rev, factors) -> tuple[str, list[dict]]:
    pre = _prereg(cfg)
    rows = comparisons(runs, cfg)
    L = ["# MNIST ↔ stylus domain gap", "",
         f"Generated from `{cfg['results_dir']}/runs.jsonl` and `reverse.jsonl` with `{cfg['_config_path']}`. "
         "PyTorch CNN (team architecture), 5 epochs, accuracy in %, mean ± std over seeds 0–4. "
         "Δ is paired by seed against the baseline (pp). p: two-sided paired t-test; p (Holm): Holm-corrected over all "
         "exploratory recipe-vs-baseline tests in both directions. The headline recipe per direction was chosen on validation "
         "data before any test run and is the confirmatory test (no correction).", ""]

    L += ["## 1. Why: measured differences (MNIST train 60,000 vs stylus 10,000)", "",
          "| Group | Measure | Unit | MNIST mean ± std | Stylus mean ± std |", "|---|---|---|---|---|"]
    for k, label, unit, grp in FACTORS:
        f = factors[k]
        L.append(f"| {grp} | {label} | {unit} | {f['mnist_mean']:.3f} ± {f['mnist_std']:.3f} | {f['stylus_mean']:.3f} ± {f['stylus_std']:.3f} |")
    hb = factors["_holes_by_class"]
    L += ["", "Share of images with a closed loop, by class:", "",
          "| | " + " | ".join(str(c) for c in range(10)) + " |", "|---|" + "---|" * 10,
          "| MNIST | " + " | ".join(f"{100 * v:.0f}%" for v in hb["mnist"]) + " |",
          "| Stylus | " + " | ".join(f"{100 * v:.0f}%" for v in hb["stylus"]) + " |", ""]

    def _dcell(r):
        return f"{r['delta']:+.1f} ± {r['delta_std']:.1f}" if 'delta' in r else '—'

    # ablation
    def row_of(d, rec):
        return next((r for r in rows if r["direction"] == d and r["recipe"] == rec), None)

    L += ["## 2. Which factors cause the drop", "", "### 2a. Stylus → MNIST: fix one mismatch at a time (stylus training images only)", "",
          "| Factor | Recipe | MNIST test acc | Δ vs baseline (pp) | p | p (Holm) |", "|---|---|---|---|---|---|"]
    b = row_of("s2m", "baseline")
    L.append(f"| none | baseline | {_fmt(b['mean'], b['std'])} | — | — | — |")
    for rec, lab in (("f_center", "position (center of mass)"), ("f_aspect", "aspect (stretch ×1.3)"), ("f_slant", "slant (random shear)"),
                     ("f_width", "stroke width (3×3 dilation)"), ("f_intensity", "intensity (peak → 1)"), ("f_soft", "soft edges (blur σ 0.6)")):
        r = row_of("s2m", rec)
        if r:
            L.append(f"| {lab} | `{rec}` | {_fmt(r['mean'], r['std'])} | {_dcell(r)} | {_p(r.get('p'))} | {_p(r.get('p_holm'))} |")
    L += ["", "### 2b. Cumulative attribution (fixed order: position → aspect → slant → width → intensity → soft edges)", "",
          "| Step | Recipe | MNIST test acc | Step gain (pp) | Cumulative gain (pp) |", "|---|---|---|---|---|"]
    prev = b["mean"]
    for rec, lab in (("f_center", "+ position"), ("c2_center_aspect", "+ aspect"), ("c3_+slant", "+ slant"),
                     ("c4_+width", "+ stroke width"), ("c5_+intensity", "+ intensity"), ("c6_+soft", "+ soft edges")):
        r = row_of("s2m", rec)
        if r:
            L.append(f"| {lab} | `{rec}` | {_fmt(r['mean'], r['std'])} | {100 * (r['mean'] - prev):+.1f} | {100 * (r['mean'] - b['mean']):+.1f} |")
            prev = r["mean"]

    if rev:
        L += ["", "### 2c. Reverse: MNIST test images degraded towards stylus style, R1 (MNIST-trained) models", "",
              "| Degradation | Accuracy | Δ vs clean (pp) | p |", "|---|---|---|---|"]
        clean = rev.get("clean")
        for name, vals in rev.items():
            if name == "clean":
                L.append(f"| none | {_fmt(np.mean(vals), np.std(vals, ddof=1))} | — | — |")
                continue
            dm, ds, p = paired(clean, vals)
            L.append(f"| {name} | {_fmt(np.mean(vals), np.std(vals, ddof=1))} | {dm:+.1f} ± {ds:.1f} | {_p(p)} |")
        L += ["", "`thin`: Zhang–Suen skeleton scaled to the stylus median ink intensity. `hard`: binarized at 0.5. "
              "`dim`: intensity × (stylus / MNIST mean ink). `uncenter`: shifted so the center-of-mass offset follows the stylus "
              "distribution. `narrow`: squeezed to a stylus aspect ratio. `upright`: deslanted (stylus digits are nearly upright). "
              "`all`: upright + narrow + thin + uncenter."]

    # fixes
    for d, title in (("s2m", "Stylus → MNIST (train 10,000 stylus, test MNIST test set)"),
                     ("m2s", "MNIST → stylus (train 50,000 MNIST, test the writers' held-out splits, mean over 5 writers)")):
        L += ["", f"## 3{'a' if d == 's2m' else 'b'}. Fix: {title}", "",
              f"Headline (pre-registered on validation data): **`{pre.get(f'headline_{d}')}`**.", "",
              "| Recipe | Selection val acc (3 seeds) | Test acc | Δ vs baseline (pp) | Seeds improved | p | p (Holm) |",
              "|---|---|---|---|---|---|---|"]
        cand = cfg["candidates"][d]
        drows = [r for r in rows if r["direction"] == d and r["recipe"] in cand]
        drows.sort(key=lambda r: -(r["select_val"] or 0))
        for r in drows:
            name = f"**`{r['recipe']}`**" if r["headline"] else f"`{r['recipe']}`"
            if r["recipe"] == "baseline":
                L.append(f"| {name} | {_fmt(r['select_val'])} | {_fmt(r['mean'], r['std'])} | — | — | — | — |")
            else:
                ph = "confirmatory" if r["headline"] else _p(r.get("p_holm"))
                L.append(f"| {name} | {_fmt(r['select_val'])} | {_fmt(r['mean'], r['std'])} | {_dcell(r)} | "
                         f"{r.get('improved', '—')}/{r.get('n_paired', '—')} | {_p(r.get('p'))} | {ph} |")
        L += ["", "Recipe definitions are in the config (`recipes`)."]

    # per class
    hs = pre.get("headline_s2m")
    L += ["", "## 4. Per class, stylus → MNIST (confusion matrices, mean ± std over seeds)", "",
          "| Recipe | recall 7 | precision 8 | predicted-8 share | 9 → 4 rate | recall 9 | precision 4 |", "|---|---|---|---|---|---|---|"]
    for rec in dict.fromkeys(["baseline", "f_width", "norm_dil_train", hs]):
        ms = _cm(runs, "s2m", rec)
        if not ms:
            continue
        cms = [class_metrics(m) for m in ms]
        cells = " | ".join(_fmt(np.mean([c[k] for c in cms]), np.std([c[k] for c in cms], ddof=1))
                           for k in ("recall_7", "precision_8", "pred_share_8", "rate_9_as_4", "recall_9", "precision_4"))
        L.append(f"| `{rec}` | {cells} |")
    return "\n".join(L) + "\n", rows


# ------------------------------------------------------------------ figures


def _style(ax):
    from digitstudy.report import _style as s

    s(ax)


def figures(cfg, runs, rev, factors, rows, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from digitstudy.report import GRID, MUTED, TEXT

    pre = _prereg(cfg)
    paths = []
    samples = np.load(Path(cfg["results_dir"]) / "factor_samples.npz")
    # 1. factor distributions
    keys = [("stroke_width", "Stroke width (px)"), ("edge_softness", "Edge softness (0 = hard)"), ("aspect", "Aspect ratio w/h"),
            ("com_offset", "Center-of-mass offset (px)"), ("slant_deg", "Slant (deg)"), ("ink_coverage", "Ink coverage (> 0.25)")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), dpi=150)
    for ax, (k, lab) in zip(axes.ravel(), keys):
        a, b_ = samples[f"mnist_{k}"], samples[f"stylus_{k}"]
        lo, hi = np.percentile(np.concatenate([a, b_]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 40)
        ax.hist(a, bins, density=True, color=COLORS[0], alpha=0.55, label="MNIST")
        ax.hist(b_, bins, density=True, color=COLORS[1], alpha=0.55, label="Stylus")
        ax.set_xlabel(lab)
        ax.set_yticks([])
        _style(ax)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Where MNIST and the stylus digits differ (per-image measures)", fontsize=10, color=TEXT)
    fig.tight_layout()
    p = out / "factor_distributions.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # 2. waterfall + individual gains
    def row(d, rec):
        return next((r for r in rows if r["direction"] == d and r["recipe"] == rec), None)

    steps = [("f_center", "position"), ("c2_center_aspect", "aspect"), ("c3_+slant", "slant"), ("c4_+width", "stroke width"),
             ("c5_+intensity", "intensity"), ("c6_+soft", "soft edges")]
    singles = [("f_center", "position"), ("f_aspect", "aspect"), ("f_slant", "slant"), ("f_width", "stroke width"),
               ("f_intensity", "intensity"), ("f_soft", "soft edges")]
    b = row("s2m", "baseline")
    if b and all(row("s2m", s) for s, _ in steps):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
        ax = axes[0]
        level = 100 * b["mean"]
        ax.bar(0, level, 0.6, color=MUTED)
        ax.text(0, level + 1, f"{level:.1f}", ha="center", fontsize=8, color=TEXT)
        for i, (rec, lab) in enumerate(steps, 1):
            new = 100 * row("s2m", rec)["mean"]
            col = COLORS[2] if new >= level else COLORS[1]
            ax.bar(i, new - level, 0.6, bottom=level, color=col)
            ax.text(i, max(new, level) + 1, f"{new - level:+.1f}", ha="center", fontsize=8, color=TEXT)
            level = new
        ax.bar(len(steps) + 1, level, 0.6, color=COLORS[0])
        ax.text(len(steps) + 1, level + 1, f"{level:.1f}", ha="center", fontsize=8, color=TEXT)
        ax.set_xticks(range(len(steps) + 2), ["baseline"] + [s[1] for s in steps] + ["all six"], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MNIST test accuracy (%)")
        ax.set_title("Cumulative (fixed order)", fontsize=9, color=TEXT)
        ax.grid(axis="y", color=GRID, lw=0.8)
        _style(ax)
        ax = axes[1]
        ds = [row("s2m", s) for s, _ in singles]
        dv = [r.get("delta", 0.0) if r else 0.0 for r in ds]
        ax.bar(range(len(singles)), dv, 0.6, yerr=[(r.get("delta_std", 0.0) if r else 0.0) for r in ds], capsize=3,
               color=[COLORS[2] if v >= 0 else COLORS[1] for v in dv], error_kw={"lw": 0.8, "ecolor": MUTED})
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xticks(range(len(singles)), [s[1] for s in singles], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Δ vs baseline (pp)")
        ax.set_title("One factor at a time", fontsize=9, color=TEXT)
        ax.grid(axis="y", color=GRID, lw=0.8)
        _style(ax)
        fig.suptitle("Stylus → MNIST: what each mismatch costs (mean ± std over 5 seeds)", fontsize=10, color=TEXT)
        fig.tight_layout()
        p = out / "factor_attribution.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    # 3. reverse degradation
    if rev:
        names = list(rev)
        fig, ax = plt.subplots(figsize=(7, 3.4), dpi=150)
        m = [100 * np.mean(rev[n]) for n in names]
        s = [100 * np.std(rev[n], ddof=1) for n in names]
        ax.bar(range(len(names)), m, 0.6, yerr=s, capsize=3, color=[MUTED] + [COLORS[1]] * (len(names) - 1),
               error_kw={"lw": 0.8, "ecolor": MUTED})
        for i, v in enumerate(m):
            ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=8, color=TEXT)
        ax.set_xticks(range(len(names)), names, fontsize=8)
        ax.set_ylabel("MNIST test accuracy (%)")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color=GRID, lw=0.8)
        _style(ax)
        ax.set_title("MNIST-trained model on MNIST test images made stylus-like", fontsize=10, color=TEXT)
        fig.tight_layout()
        p = out / "reverse_degradation.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    # 4. 9 -> 4 examples
    raw = _raw(cfg)
    hs = pre.get("headline_s2m")
    pb = Path(cfg["results_dir"]) / "preds" / "s2m__baseline.npy"
    ph = Path(cfg["results_dir"]) / "preds" / f"s2m__{hs}.npy"
    from digitstudy.gapstats import measure

    if pb.exists():
        xs, ys = raw["stylus_all"]
        xt, yt = raw["mnist_test"]
        hs_st = measure(xs[(ys == 4) | (ys == 9)])["holes"]
        st4 = xs[(ys == 4) | (ys == 9)][(ys[(ys == 4) | (ys == 9)] == 4) & (hs_st > 0)][:8]
        st9 = xs[ys == 9][:8]
        pred_b = np.load(pb)
        wrong = xt[(yt == 9) & (pred_b == 4)][:8]
        right = xt[(yt == 9) & (pred_b == 9)][:8]
        rows_img = [("Stylus 4s with a closed top", st4), ("Stylus 9s", st9),
                    ("MNIST 9s the baseline calls 4", wrong), ("MNIST 9s the baseline gets right", right)]
        if ph.exists() and hs != "baseline":
            pred_h = np.load(ph)
            rows_img.append((f"MNIST 9s still called 4 by `{hs}`", xt[(yt == 9) & (pred_h == 4)][:8]))
        fig, axes = plt.subplots(len(rows_img), 8, figsize=(8, 1.15 * len(rows_img) + 0.3), dpi=150)
        for i, (lab, imgs) in enumerate(rows_img):
            for j in range(8):
                ax = axes[i, j]
                if j < len(imgs):
                    ax.imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
            axes[i, 0].set_title(lab, fontsize=8, loc="left", color=TEXT)
        fig.tight_layout()
        p = out / "nine_vs_four_examples.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)

    # 5. examples of the preprocessing
    from digitstudy.domain_gap import datasets

    rec = cfg["recipes"].get(hs, {})
    (xs_h, _), _, (xt_h, _) = datasets(cfg, "s2m", rec, True)
    xs0, _ = raw["stylus_all"]
    xt0, _ = raw["mnist_test"]
    idx_s = np.random.default_rng(1).permutation(len(xs0))[:8]
    idx_m = np.arange(8)
    rows_img = [("MNIST test (raw)", xt0[idx_m]), (f"MNIST test as the headline model sees it", xt_h[idx_m]),
                ("Stylus (raw)", xs0[idx_s]), (f"Stylus after `{hs}` preprocessing (before augmentation)", xs_h[idx_s])]
    fig, axes = plt.subplots(4, 8, figsize=(8, 4.9), dpi=150)
    for i, (lab, imgs) in enumerate(rows_img):
        for j in range(8):
            axes[i, j].imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
        axes[i, 0].set_title(lab, fontsize=8, loc="left", color=TEXT)
    fig.tight_layout()
    p = out / "domain_examples.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)
    return paths


def build(cfg, with_figures=True):
    runs = load_runs(cfg)
    out = Path(cfg["results_dir"])
    factors = json.loads((out / "factors.json").read_text(encoding="utf-8"))
    rev = defaultdict(list)
    rp = out / "reverse.jsonl"
    if rp.exists():
        for line in rp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rev[r["variant"]].append(r["test"]["acc"])
    rev = dict(rev)
    md, rows = markdown(cfg, runs, rev, factors)
    (out / "DOMAIN_GAP.md").write_text(md, encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"comparisons": rows, "reverse": rev}, indent=2), encoding="utf-8")
    figs = []
    if with_figures:
        fd = out / "figures"
        fd.mkdir(exist_ok=True)
        figs = figures(cfg, runs, rev, factors, rows, fd)
    print(f"[domain-gap report] {len(runs)} runs -> {out.as_posix()}/DOMAIN_GAP.md, {len(figs)} figures")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/domain_gap.json")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    build(load_config(a.config), not a.no_figures)


if __name__ == "__main__":
    main()
