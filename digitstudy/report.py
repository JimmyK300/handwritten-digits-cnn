"""Aggregate results/runs.jsonl into tables (Markdown + JSON) and figures.

Statistics
----------
For every (protocol, model, condition) we compute one accuracy per seed:
  * sample_split / mnist_to_stylus: accuracy on the pooled test set.
  * lowo: the mean of the 5 held-out-writer accuracies (macro average).
Tables report mean ± sample std (ddof=1) over seeds. Deltas versus the baseline
are paired by seed (same seed, same fold) and given in percentage points (pp);
the relative change is delta / baseline accuracy. p-values come from a two-sided
paired t-test over seeds and are not corrected for multiple comparisons.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from digitstudy.study import load_runs, load_study_config

PROTOCOL_LABELS = {
    "sample_split": "Sample split (team's original)",
    "lowo": "Leave-one-writer-out",
    "mnist_to_stylus": "MNIST → stylus",
    "custom_to_mnist": "Stylus → MNIST (team's regime 4)",
}
PROTOCOL_COLORS = {"sample_split": "#2a78d6", "lowo": "#eb6834", "mnist_to_stylus": "#1baf7a", "custom_to_mnist": "#eda100"}
MODEL_COLORS = {"torch": "#2a78d6", "numpy": "#eb6834"}
MAIN_PROTOCOLS = ["sample_split", "lowo", "mnist_to_stylus"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _paired_p(deltas: list[float]) -> float | None:
    """Two-sided paired t-test p-value (H0: mean delta = 0) over seeds."""
    if len(deltas) < 2 or np.std(deltas, ddof=1) == 0:
        return None
    from scipy import stats

    return float(stats.ttest_1samp(deltas, 0.0).pvalue)


def per_seed_accuracy(runs: list[dict[str, Any]]) -> dict[tuple, dict[int, float]]:
    """(protocol, model, condition) -> {seed: accuracy}."""
    buckets: dict[tuple, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        buckets[(r["protocol"], r["model"], r["condition"])][r["seed"]].append(r["accuracy"])
    return {k: {s: float(np.mean(v)) for s, v in seeds.items()} for k, seeds in buckets.items()}


def summarize(runs: list[dict[str, Any]], baseline: str = "baseline") -> list[dict[str, Any]]:
    acc = per_seed_accuracy(runs)
    n_folds = defaultdict(set)
    for r in runs:
        n_folds[(r["protocol"], r["model"], r["condition"])].add(r["fold"])
    rows = []
    for (protocol, model, cond), seeds in sorted(acc.items()):
        vals = [seeds[s] for s in sorted(seeds)]
        row = {
            "protocol": protocol,
            "model": model,
            "condition": cond,
            "n_seeds": len(vals),
            "n_folds": len(n_folds[(protocol, model, cond)]),
            "acc_mean": float(np.mean(vals)),
            "acc_std": _std(vals),
            "per_seed": {str(s): seeds[s] for s in sorted(seeds)},
        }
        base = acc.get((protocol, model, baseline))
        if base is not None:
            common = sorted(set(seeds) & set(base))
            deltas = [100 * (seeds[s] - base[s]) for s in common]
            rel = [100 * (seeds[s] - base[s]) / base[s] for s in common]
            row.update(
                {
                    "delta_pp_mean": float(np.mean(deltas)),
                    "delta_pp_std": _std(deltas),
                    "delta_rel_pct_mean": float(np.mean(rel)),
                    "seeds_improved": int(sum(d > 0 for d in deltas)),
                    "n_paired": len(common),
                    "paired_t_p": _paired_p(deltas),
                }
            )
        rows.append(row)
    return rows


def per_writer_table(runs, protocol: str, model: str, conditions: list[str]) -> dict[str, dict[str, tuple]]:
    """writer -> condition -> (mean, std) of accuracy over seeds."""
    vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["protocol"] == protocol and r["model"] == model and r["condition"] in conditions:
            for w, a in r["per_writer_accuracy"].items():
                vals[w][r["condition"]].append(a)
    return {w: {c: (float(np.mean(v)), _std(v)) for c, v in cs.items()} for w, cs in sorted(vals.items())}


def summed_confusion(runs, protocol: str, model: str, condition: str) -> np.ndarray | None:
    cms = [np.array(r["confusion"]) for r in runs
           if r["protocol"] == protocol and r["model"] == model and r["condition"] == condition]
    return np.sum(cms, axis=0) if cms else None


def class_metrics(runs, protocol: str, model: str, condition: str, classes=(7, 8)) -> dict[str, tuple] | None:
    """Per-seed precision/recall for selected classes -> {metric: (mean, std)} in %."""
    sel = [r for r in runs if r["protocol"] == protocol and r["model"] == model and r["condition"] == condition]
    if not sel:
        return None
    vals: dict[str, list[float]] = defaultdict(list)
    for r in sel:
        cm = np.array(r["confusion"], dtype=float)
        for k in classes:
            vals[f"recall_{k}"].append(100 * cm[k, k] / max(cm[k].sum(), 1))
            vals[f"precision_{k}"].append(100 * cm[k, k] / max(cm[:, k].sum(), 1))
            vals[f"pred_share_{k}"].append(100 * cm[:, k].sum() / cm.sum())
    return {m: (float(np.mean(v)), _std(v)) for m, v in vals.items()}


# ------------------------------------------------------------------ markdown


def _fmt_acc(row) -> str:
    return f"{100 * row['acc_mean']:.2f} ± {100 * row['acc_std']:.2f}"


def _fmt_delta(row) -> str:
    if "delta_pp_mean" not in row or row["condition"] == "baseline":
        return "—"
    return (f"{row['delta_pp_mean']:+.2f} ± {row['delta_pp_std']:.2f} pp "
            f"({row['delta_rel_pct_mean']:+.1f}% rel, {row['seeds_improved']}/{row['n_paired']} seeds, "
            f"{_fmt_p(row.get('paired_t_p'))})")


def _fmt_p(p) -> str:
    return "p=n/a" if p is None else f"p={p:.3f}"


def markdown_report(cfg, rows, runs) -> str:
    baseline, primary = cfg.get("baseline_condition", "baseline"), cfg["primary_condition"]
    lines = ["# Results", "",
             f"Generated from `{Path(cfg['results_dir']).as_posix()}/runs.jsonl` ({len(runs)} runs) "
             f"with config `{cfg['_config_path']}`.", "",
             "Accuracy is mean ± std over seeds, in %. Δ is paired by seed against `baseline`, in "
             "percentage points (pp); `rel` is Δ / baseline accuracy; p is a two-sided paired t-test "
             "over seeds, not corrected for the number of conditions tested. "
             f"Pre-registered primary condition: `{primary}`.", ""]

    for model in ["torch", "numpy"]:
        mrows = [r for r in rows if r["model"] == model]
        if not mrows:
            continue
        title = "PyTorch CNN" if model == "torch" else "NumPy from-scratch CNN"
        lines += [f"## {title}", ""]
        for protocol in PROTOCOL_LABELS:
            prow = [r for r in mrows if r["protocol"] == protocol]
            if not prow:
                continue
            order = list(cfg["conditions"])
            prow.sort(key=lambda r: order.index(r["condition"]) if r["condition"] in order else 99)
            n = prow[0]
            folds = f", {n['n_folds']} held-out writers" if protocol == "lowo" else ""
            lines += [f"### {PROTOCOL_LABELS[protocol]} ({n['n_seeds']} seeds{folds})", "",
                      "| Condition | Accuracy (%) | Δ vs baseline |", "|---|---|---|"]
            for r in prow:
                name = f"**{r['condition']}**" if r["condition"] in (primary, baseline) else r["condition"]
                lines.append(f"| {name} | {_fmt_acc(r)} | {_fmt_delta(r)} |")
            lines.append("")

    c2m = cfg.get("protocol_conditions", {}).get("custom_to_mnist", [])
    if c2m and any(r["protocol"] == "custom_to_mnist" for r in runs):
        lines += ["## Stylus → MNIST: the class-7 / class-8 failure", "",
                  "Per-seed metrics from each run's confusion matrix, mean ± std over seeds, in %. "
                  "`pred share 8` is the fraction of all MNIST test images predicted as 8 "
                  "(the true share of 8s is 9.7%).", ""]
        for model in ["torch", "numpy"]:
            mlines = []
            for cond in c2m:
                m = class_metrics(runs, "custom_to_mnist", model, cond)
                if m:
                    cells = " | ".join(f"{m[k][0]:.1f} ± {m[k][1]:.1f}"
                                       for k in ["recall_7", "precision_8", "recall_8", "pred_share_8"])
                    mlines.append(f"| {cond} | {cells} |")
            if mlines:
                title = "PyTorch CNN" if model == "torch" else "NumPy CNN"
                lines += [f"### {title}", "", "| Condition | recall 7 | precision 8 | recall 8 | pred share 8 |",
                          "|---|---|---|---|---|", *mlines, ""]

    pw = per_writer_table(runs, "lowo", "torch", [baseline, primary])
    if pw:
        lines += ["## Per held-out writer (PyTorch, leave-one-writer-out)", "",
                  f"| Held-out writer | {baseline} (%) | {primary} (%) | Δ (pp) |", "|---|---|---|---|"]
        for w, cs in pw.items():
            if baseline in cs and primary in cs:
                b, p = cs[baseline], cs[primary]
                lines.append(f"| {w} | {100*b[0]:.2f} ± {100*b[1]:.2f} | {100*p[0]:.2f} ± {100*p[1]:.2f} | "
                             f"{100*(p[0]-b[0]):+.2f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ figures


def _style(ax):
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)


def plot_deltas(cfg, rows, out: Path, model: str = "torch") -> Path | None:
    import matplotlib.pyplot as plt

    conds = [c for c in cfg["conditions"]
             if c != cfg.get("baseline_condition", "baseline") and not cfg["conditions"][c].get("protocols")]
    protocols = [p for p in MAIN_PROTOCOLS if any(r["protocol"] == p and r["model"] == model for r in rows)]
    if not protocols:
        return None
    lookup = {(r["protocol"], r["condition"]): r for r in rows if r["model"] == model}
    fig, ax = plt.subplots(figsize=(10, 0.42 * len(conds) * len(protocols) + 1.2), dpi=150)
    height = 0.8 / len(protocols)
    y = np.arange(len(conds))
    for i, p in enumerate(protocols):
        means = [lookup.get((p, c), {}).get("delta_pp_mean", np.nan) for c in conds]
        stds = [lookup.get((p, c), {}).get("delta_pp_std", 0) for c in conds]
        pos = y + (i - (len(protocols) - 1) / 2) * height
        ax.barh(pos, means, height=height * 0.88, xerr=stds, color=PROTOCOL_COLORS[p],
                error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2}, label=PROTOCOL_LABELS[p])
        for yy, m, s in zip(pos, means, stds):
            if np.isfinite(m):
                x = m + (s + 0.4) * (1 if m >= 0 else -1)
                ax.text(x, yy, f"{m:+.1f}", va="center", ha="left" if m >= 0 else "right", fontsize=7, color=TEXT)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, conds)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy change vs baseline (percentage points, mean ± std over seeds)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title(f"Effect of each preprocessing condition ({'PyTorch' if model == 'torch' else 'NumPy'} CNN)",
                 loc="left", color=TEXT, fontsize=11)
    fig.tight_layout()
    path = out / f"deltas_{model}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_custom_to_mnist(cfg, rows, out: Path) -> Path | None:
    import matplotlib.pyplot as plt

    base = cfg.get("baseline_condition", "baseline")
    conds = [c for c in cfg.get("protocol_conditions", {}).get("custom_to_mnist", []) if c != base]
    lookup = {(r["model"], r["condition"]): r for r in rows if r["protocol"] == "custom_to_mnist"}
    models = [m for m in ["torch", "numpy"] if any(k[0] == m for k in lookup)]
    if not conds or not models:
        return None
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(conds) + 1.4), dpi=150)
    height = 0.8 / len(models)
    y = np.arange(len(conds))
    for i, model in enumerate(models):
        pos = y + (i - (len(models) - 1) / 2) * height
        rs = [lookup.get((model, c)) for c in conds]
        keep = [j for j, r in enumerate(rs) if r is not None and "delta_pp_mean" in r]
        means = np.array([rs[j]["delta_pp_mean"] for j in keep])
        stds = np.array([rs[j]["delta_pp_std"] for j in keep])
        ax.barh(pos[keep], means, height=height * 0.88, xerr=stds, color=MODEL_COLORS[model],
                error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2},
                label="PyTorch CNN" if model == "torch" else "NumPy CNN")
        for yy, m, sd in zip(pos[keep], means, stds):
            ax.text(m + (sd + 0.5) * (1 if m >= 0 else -1), yy, f"{m:+.1f}", va="center",
                    ha="left" if m >= 0 else "right", fontsize=7, color=TEXT)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, conds)
    ax.invert_yaxis()
    ax.set_xlabel("MNIST test accuracy change vs baseline (pp, mean ± std over seeds)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    _style(ax)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_title("Stylus → MNIST: preprocessing applied to the stylus training images",
                 loc="left", color=TEXT, fontsize=11)
    fig.tight_layout()
    path = out / "deltas_custom_to_mnist.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_confusions(cfg, runs, out: Path, model: str = "torch", protocols=("lowo", "mnist_to_stylus"),
                    conditions=None, name: str | None = None) -> Path | None:
    import matplotlib.pyplot as plt

    baseline, primary = cfg.get("baseline_condition", "baseline"), cfg["primary_condition"]
    conditions = conditions or [baseline, primary]
    panels = []
    for protocol in protocols:
        for cond in conditions:
            cm = summed_confusion(runs, protocol, model, cond)
            if cm is not None:
                panels.append((protocol, cond, cm))
    if not panels:
        return None
    cols = 2
    rows_n = (len(panels) + 1) // 2
    fig, axes = plt.subplots(rows_n, cols, figsize=(9, 4.3 * rows_n), dpi=150, squeeze=False)
    for ax, (protocol, cond, cm) in zip(axes.flat, panels):
        norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(10):
            for j in range(10):
                v = norm[i, j]
                if v >= 0.02:
                    ax.text(j, i, f"{100*v:.0f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v > 0.6 else TEXT)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{PROTOCOL_LABELS[protocol]}\n{cond}  (acc {100*acc:.1f}%)", fontsize=9, color=TEXT, loc="left")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("True", fontsize=8)
        ax.tick_params(labelsize=7, colors=MUTED)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle("Row-normalised confusion matrices (%), summed over seeds and folds", fontsize=10, color=TEXT)
    fig.tight_layout()
    path = out / (name or f"confusion_{model}.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_examples(cfg, out: Path) -> Path:
    import matplotlib.pyplot as plt

    from digitstudy import data as D
    from digitstudy.transforms import apply_preprocess

    writers = D.load_stylus()
    xm, ym = D.load_mnist("t10k")
    shown = ["baseline", "dilate_k2_i1", "dilate_k3_i1", "dilate_k3_i2", "gain_1.5", "norm_mnist",
             "norm_mnist_dilate_k3_i1"]
    shown = [c for c in shown if c in cfg["conditions"]]
    names = list(writers)
    digits = [3, 7, 8]
    fig, axes = plt.subplots(len(digits), len(shown) + 1, figsize=(1.5 * (len(shown) + 1), 1.6 * len(digits)), dpi=150)
    for r, d in enumerate(digits):
        w = writers[names[r % len(names)]]
        img = w.images[np.where(w.labels == d)[0][0]]
        axes[r, 0].imshow(xm[np.where(ym == d)[0][0]], cmap="gray", vmin=0, vmax=1)
        if r == 0:
            axes[r, 0].set_title("MNIST\n(reference)", fontsize=7, color=TEXT)
        for c, cond in enumerate(shown, 1):
            axes[r, c].imshow(apply_preprocess(img[None], cfg["conditions"][cond].get("preprocess"))[0],
                              cmap="gray", vmin=0, vmax=1)
            if r == 0:
                axes[r, c].set_title(f"stylus\n{cond}", fontsize=7, color=TEXT)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    path = out / "examples.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def build_report(cfg, figures: bool = True) -> dict[str, Any]:
    out_dir = Path(cfg["results_dir"])
    runs = load_runs(out_dir)
    if not runs:
        raise SystemExit(f"No runs in {out_dir.as_posix()}/runs.jsonl; run the study first.")
    rows = summarize(runs, cfg.get("baseline_condition", "baseline"))
    (out_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (out_dir / "RESULTS.md").write_text(markdown_report(cfg, rows, runs), encoding="utf-8")
    paths = []
    if figures:
        import matplotlib

        matplotlib.use("Agg")
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        for model in ["torch", "numpy"]:
            paths += [plot_deltas(cfg, rows, fig_dir, model), plot_confusions(cfg, runs, fig_dir, model)]
            c2m_all = cfg.get("protocol_conditions", {}).get("custom_to_mnist", [])
            c2m_conds = [c for c in ["baseline", cfg["primary_condition"], "norm_mnist", "norm_mnist_dilate_k3_i1"]
                         if c in c2m_all]
            if c2m_conds:
                paths.append(plot_confusions(cfg, runs, fig_dir, model, ("custom_to_mnist",), c2m_conds,
                                             f"confusion_custom_to_mnist_{model}.png"))
        paths.append(plot_custom_to_mnist(cfg, rows, fig_dir))
        paths.append(plot_examples(cfg, fig_dir))
    print(f"[report] wrote {out_dir.as_posix()}/RESULTS.md, summary.json and {len([p for p in paths if p])} figures")
    return {"rows": rows, "figures": [str(p) for p in paths if p]}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Summarize study runs.")
    parser.add_argument("--config", default="configs/study.json")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    build_report(load_study_config(args.config), figures=not args.no_figures)


if __name__ == "__main__":
    main()
