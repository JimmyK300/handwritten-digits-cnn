"""Tables and figures for the five regimes (results/regimes/runs.jsonl -> REGIMES.md).

Definitions (all accuracies on held-out data only):
  seen acc        mean accuracy over the training writers' held-out splits
                  (R1: the MNIST test set).
  unseen acc      mean accuracy over writers not used for training.
  WriterBias      max(A_i) - min(A_i) over the 5 writers' held-out splits, as in
                  the proposal. WriterBias(unseen) uses unseen writers only.
  gap acc         train accuracy - pooled seen held-out accuracy (pp).
  gap loss        pooled seen held-out loss - train loss (positive = overfitting).
  cross gap       seen acc - unseen acc (pp): the cross-writer generalization gap.
  forgetting      R1 MNIST accuracy - R5 MNIST accuracy, same seed (pp).
Each statistic is averaged over writer combinations within a seed, then reported
as mean ± std over the 5 seeds.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from digitstudy.regimes import load_config, load_runs

WRITERS_KEY = "mnist"


def _std(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def prf(cm: np.ndarray) -> dict[str, Any]:
    cm = np.asarray(cm, dtype=float)
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(10), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(10), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(10), where=(prec + rec) > 0)
    return {"precision": prec, "recall": rec, "f1": f1,
            "macro_precision": float(prec.mean()), "macro_recall": float(rec.mean()), "macro_f1": float(f1.mean())}


def group_of(r) -> str:
    k = len(r["train_writers"])
    if r["regime"] in ("R3f", "R4f", "R3fe", "R4fe"):
        base = "R3 k=3" if r["regime"].startswith("R3") else "R4"
        eq = " eq" if r["regime"].endswith("e") else ""
        return f"{base} @{int(round(100 * r['fraction']))}%{eq}"
    if r["regime"] == "R5s":
        who = "1 writer" if k == 1 else f"{k} writers"
        return f"R5 ({who}) lr×{r['lr_scale']:g} ep{r['epochs']}"
    if r["regime"] == "R1":
        return "R1"
    if r["regime"] == "R1m":
        return f"R1m n={r['n_train']}"
    if r["regime"] == "R5":
        return "R5 (1 writer)" if k == 1 else f"R5 ({k} writers)"
    if r["regime"] == "R3s":
        return f"R3s k={k}"
    if r["regime"] == "R3":
        return f"R3 k={k}"
    return r["regime"]


def derive(r: dict[str, Any], r1_mnist: dict[int, float]) -> dict[str, Any]:
    tests = r["tests"]
    writers = sorted(k for k in tests if k != WRITERS_KEY)
    acc = {w: tests[w]["acc"] for w in writers}
    seen = set(r["train_writers"])
    unseen = [w for w in writers if w not in seen]
    d: dict[str, Any] = {"mnist_acc": tests["mnist"]["acc"], "train_acc": r["train_acc"], "n_train": r["n_train"]}
    d["writer_bias"] = max(acc.values()) - min(acc.values())
    d["all_writers_acc"] = float(np.mean(list(acc.values())))
    # composite test: MNIST test set + all 5 writers' held-out splits
    n_s = sum(tests[w]["n"] for w in writers)
    c_s = sum(tests[w]["acc"] * tests[w]["n"] for w in writers)
    d["stylus_pooled_acc"] = c_s / n_s
    d["composite_balanced"] = (tests["mnist"]["acc"] + c_s / n_s) / 2
    d["composite_pooled"] = (c_s + tests["mnist"]["acc"] * tests["mnist"]["n"]) / (n_s + tests["mnist"]["n"])
    if r["regime"] in ("R1", "R1m"):
        seen_acc, seen_loss = tests["mnist"]["acc"], tests["mnist"]["loss"]
        d["seen_acc"] = seen_acc
        unseen = writers
    else:
        n = sum(tests[w]["n"] for w in seen)
        seen_acc = sum(tests[w]["acc"] * tests[w]["n"] for w in seen) / n
        seen_loss = sum(tests[w]["loss"] * tests[w]["n"] for w in seen) / n
        d["seen_acc"] = float(np.mean([acc[w] for w in seen]))
    d["gap_acc"] = r["train_acc"] - seen_acc
    d["gap_loss"] = seen_loss - r["train_loss"]
    if unseen:
        d["unseen_acc"] = float(np.mean([acc[w] for w in unseen]))
        if r["regime"] not in ("R1", "R1m"):
            d["cross_gap"] = d["seen_acc"] - d["unseen_acc"]
        cm = sum(np.array(tests[w]["confusion"]) for w in unseen)
        d["unseen_macro_f1"] = prf(cm)["macro_f1"]
        if len(unseen) >= 2:
            d["writer_bias_unseen"] = max(acc[w] for w in unseen) - min(acc[w] for w in unseen)
    d["mnist_macro_f1"] = prf(np.array(tests["mnist"]["confusion"]))["macro_f1"]
    if r["regime"] in ("R5", "R5s") and r["seed"] in r1_mnist:
        d["forgetting"] = r1_mnist[r["seed"]] - tests["mnist"]["acc"]
    if r.get("curve"):
        c = {p["epoch"]: p for p in r["curve"]}
        for e in (0, 1):
            if e in c and c[e].get("unseen_acc") is not None:
                d[f"unseen_acc_ep{e}"] = c[e]["unseen_acc"]
            if e in c:
                d[f"mnist_acc_ep{e}"] = c[e]["mnist_acc"]
    return d


def aggregate(runs) -> dict[tuple[str, str], dict[str, tuple[float, float, list[float]]]]:
    """(group, arm) -> metric -> (mean, std, per-seed values)."""
    r1_mnist = {r["seed"]: r["tests"]["mnist"]["acc"] for r in runs if r["regime"] == "R1" and r["arm"] == "baseline"}
    per: dict[tuple, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        for m, v in derive(r, r1_mnist).items():
            per[(group_of(r), r["arm"])][m][r["seed"]].append(v)
    out = {}
    for g, metrics in per.items():
        out[g] = {}
        for m, seeds in metrics.items():
            vals = [float(np.mean(seeds[s])) for s in sorted(seeds)]
            out[g][m] = (float(np.mean(vals)), _std(vals), vals)
    return out


def _pct(stat, digits=1, scale=100.0):
    if stat is None:
        return "—"
    return f"{scale * stat[0]:.{digits}f} ± {scale * stat[1]:.{digits}f}"


def _paired(a, b):
    """mean±std of (b - a) in pp, and paired t-test p."""
    from scipy import stats

    d = 100 * (np.array(b) - np.array(a))
    p = float(stats.ttest_1samp(d, 0).pvalue) if len(d) > 1 and d.std() > 0 else float("nan")
    return float(d.mean()), _std(list(d)), p


GROUP_ORDER = ["R1", "R2", "R3 k=2", "R3 k=3", "R3 k=4", "R4", "R3s k=2", "R3s k=3", "R3s k=4",
               "R5 (1 writer)", "R5 (3 writers)", "R1m n=1712", "R1m n=3424", "R1m n=5136", "R1m n=6848"]


def markdown(cfg, runs, agg) -> str:
    arms = list(cfg["arms"])
    L = ["# Regimes R1–R5", "",
         f"Generated from `{cfg['results_dir']}/runs.jsonl` ({len(runs)} runs) with `{cfg['_config_path']}`. "
         "All values are mean ± std over 5 seeds (each seed first averaged over writer combinations), in %, "
         "except gap loss (nats). Test sets: each writer's held-out split (286 images) and the MNIST test set "
         "(10,000). Definitions are in `digitstudy/regime_report.py`.", ""]

    def g(group, arm="baseline"):
        return agg.get((group, arm), {})

    L += ["## 1. All regimes (baseline arm, no preprocessing)", "",
          "| Regime | Train images | Train acc | Seen held-out acc | Unseen-writer acc | MNIST acc | WriterBias (5 writers) | WriterBias (unseen) | Gap acc (pp) | Gap loss | Cross-writer gap (pp) | Unseen macro-F1 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for grp in GROUP_ORDER:
        a = g(grp)
        if not a:
            continue
        n = int(round(a["n_train"][0]))
        L.append(f"| {grp} | {n} | {_pct(a['train_acc'])} | {_pct(a['seen_acc'])} | {_pct(a.get('unseen_acc'))} | "
                 f"{_pct(a['mnist_acc'])} | {_pct(a['writer_bias'])} | {_pct(a.get('writer_bias_unseen'))} | "
                 f"{_pct(a['gap_acc'])} | {_pct(a['gap_loss'], 3, 1.0)} | {_pct(a.get('cross_gap'))} | {_pct(a.get('unseen_macro_f1'))} |")
    L += ["", "R1's seen held-out accuracy is MNIST test accuracy and its unseen-writer accuracy covers all 5 writers. "
          "R4 has no unseen writer. R3s caps the total training set at one writer's size (≈1712 images) for every k.", ""]

    # composite
    comp_groups = ["R1", "R1m n=1712", "R1m n=3424", "R1m n=5136", "R1m n=6848", "R2", "R3s k=4", "R3 k=2", "R3 k=3",
                   "R3 k=4", "R4", "R5 (1 writer)", "R5 (3 writers)"]
    if g("R1m n=1712"):
        L += ["## 1b. Composite test: MNIST test set + all 5 writers (baseline arm; added 2026-11-06, exploratory)", "",
              "Every model scored on both domains at once: the 10,000 MNIST test images plus the 1,434 held-out images of all 5 "
              "writers. *Balanced* = mean of MNIST accuracy and pooled stylus accuracy (each domain weighted equally). *Pooled* = "
              "accuracy over all 11,434 images, so MNIST is 87% of the weight. For writer-trained models the stylus column "
              "includes their own writers' held-out splits.", "",
              "| Regime | Train images | MNIST acc | Stylus acc (5 writers) | Balanced | Pooled |", "|---|---|---|---|---|---|"]
        for grp in comp_groups:
            a = g(grp)
            if a and "composite_balanced" in a:
                L.append(f"| {grp} | {int(round(a['n_train'][0]))} | {_pct(a['mnist_acc'])} | {_pct(a['stylus_pooled_acc'])} | "
                         f"**{_pct(a['composite_balanced'])}** | {_pct(a['composite_pooled'])} |")
        L.append("")
        for lo, hi in (("R2", "R1m n=1712"), ("R3s k=4", "R1m n=1712"), ("R3 k=4", "R1m n=6848"), ("R4", "R1m n=6848"),
                       ("R4", "R1"), ("R1", "R5 (3 writers)"), ("R4", "R5 (3 writers)")):
            if g(lo) and g(hi):
                parts = []
                for m in ("composite_balanced", "composite_pooled"):
                    dm, ds, pv = _paired(g(lo)[m][2], g(hi)[m][2])
                    parts.append(f"{m.split('_')[1]} {dm:+.1f} ± {ds:.1f} pp (p = {pv:.2g})")
                L.append(f"- {hi} vs {lo}: " + "; ".join(parts) + ".")
        L.append("")

    # diversity
    L += ["## 2. Core hypothesis: writer diversity (baseline arm)", ""]
    rows = []
    for series, fmt in (("full data", lambda k: "R2" if k == 1 else f"R3 k={k}"),
                        ("size-matched", lambda k: "R2" if k == 1 else f"R3s k={k}")):
        for k in (1, 2, 3, 4):
            a = g(fmt(k))
            if a:
                rows.append((series, k, a))
    L += ["| Series | Writers k | Train images | Unseen-writer acc | WriterBias (5 writers) | WriterBias (unseen) | Gap acc (pp) | Cross-writer gap (pp) | MNIST acc |",
          "|---|---|---|---|---|---|---|---|---|"]
    for series, k, a in rows:
        L.append(f"| {series} | {k} | {int(round(a['n_train'][0]))} | {_pct(a.get('unseen_acc'))} | "
                 f"{_pct(a['writer_bias'])} | {_pct(a.get('writer_bias_unseen'))} | {_pct(a['gap_acc'])} | "
                 f"{_pct(a.get('cross_gap'))} | {_pct(a['mnist_acc'])} |")
    L.append("")
    tests = diversity_tests(agg)
    for t in tests:
        L.append(f"- {t}")
    L.append("")

    # size sweep
    fr = [int(round(100 * f)) for f in cfg.get("size_fractions", [])]
    if fr:
        L += ["## 2b. Training-set size sweep (RQ2, baseline arm)", "",
              "| Regime | Share of each writer's train split | Train images | Gap acc (pp) | Gap loss | Seen held-out acc | Unseen-writer acc | WriterBias (5 writers) | MNIST acc |",
              "|---|---|---|---|---|---|---|---|---|"]
        for base in ("R3 k=3", "R4"):
            for pct in fr + [100]:
                a = g(base if pct == 100 else f"{base} @{pct}%")
                if not a:
                    continue
                L.append(f"| {base} | {pct}% | {int(round(a['n_train'][0]))} | {_pct(a['gap_acc'])} | "
                         f"{_pct(a['gap_loss'], 3, 1.0)} | {_pct(a['seen_acc'])} | {_pct(a.get('unseen_acc'))} | "
                         f"{_pct(a['writer_bias'])} | {_pct(a['mnist_acc'])} |")
        L.append("")
        for line in size_tests(agg, fr):
            L.append(f"- {line}")
        L.append("")

    # arms
    L += ["## 3. Preprocessing arms", "",
          "Unseen-writer accuracy (MNIST accuracy for the R4 row), Δ vs baseline paired by seed (pp, p from paired t-test).", ""]
    L += ["| Regime | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for grp in GROUP_ORDER:
        metric = "mnist_acc" if grp == "R4" else "unseen_acc"
        base = g(grp, "baseline").get(metric)
        if base is None:
            continue
        cells = []
        for arm in arms:
            s = g(grp, arm).get(metric)
            if s is None:
                cells.append("—")
            elif arm == "baseline":
                cells.append(_pct(s))
            else:
                dm, ds, p = _paired(base[2], s[2])
                cells.append(f"{_pct(s)} ({dm:+.1f}, p={p:.2f})")
        L.append(f"| {grp} | " + " | ".join(cells) + " |")
    L.append("")
    L += ["MNIST accuracy for models trained on stylus data, by arm:", "",
          "| Regime | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for grp in ["R2", "R3 k=3", "R4"]:
        base = g(grp, "baseline").get("mnist_acc")
        if base is None:
            continue
        cells = []
        for arm in arms:
            s = g(grp, arm).get("mnist_acc")
            if s is None:
                cells.append("—")
            elif arm == "baseline":
                cells.append(_pct(s))
            else:
                dm, ds, p = _paired(base[2], s[2])
                cells.append(f"{_pct(s)} ({dm:+.1f}, p={p:.2f})")
        L.append(f"| {grp} | " + " | ".join(cells) + " |")
    L.append("")

    # equal-steps size sweep
    efr = [int(round(100 * f)) for f in cfg.get("equal_steps_fractions", [])]
    if efr and g(f"R4 @{efr[0]}% eq"):
        L += ["## 2c. Size sweep with equal optimizer steps (RQ2 without the underfitting confound)", "",
              f"Epochs = round({cfg['training']['epochs']} / share), so every row takes the optimizer steps of the 100% run.", "",
              "| Regime | Share | Train images | Epochs | Train acc | Gap acc (pp) | Gap loss | Seen held-out acc | Unseen-writer acc | WriterBias (5 writers) | MNIST acc |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for base in ("R3 k=3", "R4"):
            for pct in efr + [100]:
                a = g(base if pct == 100 else f"{base} @{pct}% eq")
                if not a:
                    continue
                ep = int(round(cfg["training"]["epochs"] * 100 / pct))
                L.append(f"| {base} | {pct}% | {int(round(a['n_train'][0]))} | {ep} | {_pct(a['train_acc'])} | {_pct(a['gap_acc'])} | "
                         f"{_pct(a['gap_loss'], 3, 1.0)} | {_pct(a['seen_acc'])} | {_pct(a.get('unseen_acc'))} | "
                         f"{_pct(a['writer_bias'])} | {_pct(a['mnist_acc'])} |")
        L.append("")
        for t in size_tests(agg, efr, " eq"):
            L.append(f"- {t}")
        L.append("")

    # R5
    L += ["## 4. R5: MNIST pre-training + fine-tuning (lr × 0.1)", "",
          "| Arm | Data | Init | Same/seen-writer acc | Unseen-writer acc | MNIST acc | Forgetting vs R1 (pp) | Unseen acc after 1 epoch |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in cfg.get("finetune_arms", arms):
        for scratch, ft, label in (("R2", "R5 (1 writer)", "1 writer"), (f"R3 k={cfg.get('finetune_k', 3)}",
                                                                         f"R5 ({cfg.get('finetune_k', 3)} writers)", f"{cfg.get('finetune_k', 3)} writers")):
            for grp, init in ((scratch, "scratch"), (ft, "MNIST")):
                a = g(grp, arm)
                if not a:
                    continue
                L.append(f"| {arm} | {label} | {init} | {_pct(a['seen_acc'])} | {_pct(a.get('unseen_acc'))} | "
                         f"{_pct(a['mnist_acc'])} | {_pct(a.get('forgetting'))} | {_pct(a.get('unseen_acc_ep1'))} |")
    r1 = g("R1")
    if r1:
        L += ["", f"R1 reference (MNIST-trained, before fine-tuning): MNIST {_pct(r1['mnist_acc'])}, "
              f"all-writer {_pct(r1['all_writers_acc'])}.", ""]
    for line in r5_tests(agg, cfg):
        L.append(f"- {line}")
    L.append("")

    # R5 sweep
    sw = cfg.get("finetune_sweep")
    if sw:
        k = cfg.get("finetune_k", 3)
        L += ["## 4b. R5 fine-tuning sweep: learning rate × epochs (baseline arm)", "",
              "Gain = unseen-writer accuracy minus the scratch model on the same data (R2 or R3), paired by seed. "
              "Forgetting = R1's MNIST accuracy minus the fine-tuned model's, same seed.", "",
              "| Data | lr scale | Epochs | Unseen-writer acc | Gain vs scratch (pp) | p (gain) | MNIST acc | Forgetting (pp) | p (forgetting > 0) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for who, scratch in (("1 writer", "R2"), (f"{k} writers", f"R3 k={k}")):
            for lr_s in sw["lr_scales"]:
                for ep in sw["epochs"]:
                    grp = r5_group(cfg, who, lr_s, ep)
                    a, sc = g(grp), g(scratch)
                    if not a or "unseen_acc" not in a:
                        continue
                    dm, ds, p = _paired(sc["unseen_acc"][2], a["unseen_acc"][2])
                    fg = a.get("forgetting")
                    pf = _paired([0.0] * len(fg[2]), fg[2])[2] if fg else float("nan")
                    L.append(f"| {who} | ×{lr_s:g} | {ep} | {_pct(a['unseen_acc'])} | {dm:+.1f} ± {ds:.1f} | {p:.2g} | "
                             f"{_pct(a['mnist_acc'])} | {_pct(fg)} | {pf:.2g} |")
        L.append("")

    # per-class
    L += ["## 5. Per-class precision / recall / F1 (baseline arm, confusion matrices summed over seeds and combinations)", ""]
    for title, sel, which in (
        ("R1 (MNIST-trained) on all 5 writers", lambda r: r["regime"] == "R1", "writers"),
        ("R2 (single writer) on unseen writers", lambda r: r["regime"] == "R2", "unseen"),
        ("R4 (all writers) on the MNIST test set", lambda r: r["regime"] == "R4", "mnist"),
    ):
        cm = np.zeros((10, 10))
        for r in runs:
            if r["arm"] != "baseline" or not sel(r):
                continue
            if which == "mnist":
                cm += np.array(r["tests"]["mnist"]["confusion"])
            else:
                for w, t in r["tests"].items():
                    if w == "mnist" or (which == "unseen" and w in r["train_writers"]):
                        continue
                    cm += np.array(t["confusion"])
        if cm.sum() == 0:
            continue
        m = prf(cm)
        L += [f"### {title}", "", "| Class | Precision | Recall | F1 |", "|---|---|---|---|"]
        for c in range(10):
            L.append(f"| {c} | {100*m['precision'][c]:.1f} | {100*m['recall'][c]:.1f} | {100*m['f1'][c]:.1f} |")
        L += [f"| **Macro avg** | {100*m['macro_precision']:.1f} | {100*m['macro_recall']:.1f} | {100*m['macro_f1']:.1f} |", ""]

    # per-writer
    L += ["## 6. Per-writer held-out accuracy (baseline arm)", "",
          "| Writer | R1 (MNIST-trained) | R2 own writer | R2 as unseen (mean over other trainers) | R4 (all writers) |",
          "|---|---|---|---|---|"]
    for w in sorted({k for r in runs for k in r["tests"] if k != "mnist"}):
        def mean_acc(sel):
            v = defaultdict(list)
            for r in runs:
                if r["arm"] == "baseline" and sel(r):
                    v[r["seed"]].append(r["tests"][w]["acc"])
            vals = [np.mean(x) for x in v.values()]
            return (float(np.mean(vals)), _std(vals)) if vals else None
        L.append(f"| {w} | {_pct(mean_acc(lambda r: r['regime'] == 'R1'))} | "
                 f"{_pct(mean_acc(lambda r: r['regime'] == 'R2' and r['train_writers'] == [w]))} | "
                 f"{_pct(mean_acc(lambda r: r['regime'] == 'R2' and r['train_writers'] != [w]))} | "
                 f"{_pct(mean_acc(lambda r: r['regime'] == 'R4'))} |")
    L.append("")
    return "\n".join(L) + "\n"


def r5_group(cfg, who: str, lr_s: float, ep: int) -> str:
    if lr_s == float(cfg.get("finetune_lr_scale", 0.1)) and ep == cfg["training"]["epochs"]:
        return f"R5 ({who})"
    return f"R5 ({who}) lr×{lr_s:g} ep{ep}"


def diversity_tests(agg) -> list[str]:
    from scipy import stats

    out = []
    for series, fmt in (("full data", lambda k: "R2" if k == 1 else f"R3 k={k}"),
                        ("size-matched", lambda k: "R2" if k == 1 else f"R3s k={k}")):
        for metric, label in (("unseen_acc", "unseen-writer accuracy"), ("writer_bias", "WriterBias (5 writers)"),
                              ("gap_acc", "gap acc"), ("cross_gap", "cross-writer gap")):
            ks, vs, ends = [], [], {}
            for k in (1, 2, 3, 4):
                s = agg.get((fmt(k), "baseline"), {}).get(metric)
                if s:
                    ks += [k] * len(s[2])
                    vs += s[2]
                    ends[k] = s[2]
            if len(set(ks)) < 2:
                continue
            rho, p_rho = stats.spearmanr(ks, vs)
            line = f"**{series}, {label}**: Spearman ρ(k, value) = {rho:+.2f} (p = {p_rho:.3g}, n = {len(vs)} seed-level points)"
            if 1 in ends and 4 in ends:
                dm, ds, p = _paired(ends[1], ends[4])
                line += f"; k=4 vs k=1: {dm:+.1f} ± {ds:.1f} pp (paired t, p = {p:.3g})"
            out.append(line + ".")
    return out


def size_tests(agg, fractions, suffix: str = "") -> list[str]:
    from scipy import stats

    out = []
    for base in ("R3 k=3", "R4"):
        for metric, label in (("gap_acc", "gap acc"), ("gap_loss", "gap loss"), ("writer_bias", "WriterBias")):
            xs, vs, ends = [], [], {}
            for pct in fractions + [100]:
                s = agg.get((base if pct == 100 else f"{base} @{pct}%{suffix}", "baseline"), {}).get(metric)
                if s:
                    xs += [pct] * len(s[2])
                    vs += s[2]
                    ends[pct] = s[2]
            if len(set(xs)) < 2:
                continue
            rho, p = stats.spearmanr(xs, vs)
            lo = min(ends)
            line = f"**{base}, {label}**: Spearman ρ(size, value) = {rho:+.2f} (p = {p:.3g})"
            if 100 in ends and lo != 100:
                scale = 1.0 if metric == "gap_loss" else 100.0
                d = scale * (np.array(ends[100]) - np.array(ends[lo]))
                tp = float(stats.ttest_1samp(d, 0).pvalue) if d.std() > 0 else float("nan")
                unit = "nats" if metric == "gap_loss" else "pp"
                line += f"; 100% vs {lo}%: {d.mean():+.3g} ± {_std(list(d)):.2g} {unit} (paired t, p = {tp:.3g})"
            out.append(line + ".")
    return out


def r5_tests(agg, cfg) -> list[str]:
    out = []
    k = cfg.get("finetune_k", 3)
    for scratch, ft in (("R2", "R5 (1 writer)"), (f"R3 k={k}", f"R5 ({k} writers)")):
        a, b = agg.get((scratch, "baseline"), {}), agg.get((ft, "baseline"), {})
        for metric, label in (("unseen_acc", "unseen-writer acc"), ("mnist_acc", "MNIST acc"), ("writer_bias", "WriterBias")):
            if metric in a and metric in b:
                dm, ds, p = _paired(a[metric][2], b[metric][2])
                out.append(f"{ft} vs {scratch} (baseline arm), {label}: {dm:+.1f} ± {ds:.1f} pp (p = {p:.3g}).")
    return out


# ------------------------------------------------------------------ figures


def figures(cfg, agg, runs, out: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from digitstudy.report import GRID, MUTED, TEXT, _style

    paths = []
    colors = {"full data": "#2a78d6", "size-matched": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ax, metric, ylabel in ((axes[0], "unseen_acc", "Unseen-writer accuracy (%)"),
                               (axes[1], "writer_bias", "WriterBias, 5 writers (pp)")):
        for series, fmt in (("full data", lambda k: "R2" if k == 1 else f"R3 k={k}"),
                            ("size-matched", lambda k: "R2" if k == 1 else f"R3s k={k}")):
            ks, m, s = [], [], []
            for k in (1, 2, 3, 4):
                st = agg.get((fmt(k), "baseline"), {}).get(metric)
                if st:
                    ks.append(k)
                    m.append(100 * st[0])
                    s.append(100 * st[1])
            if ks:
                ax.errorbar(ks, m, yerr=s, marker="o", ms=6, lw=2, capsize=3, color=colors[series], label=series)
                ax.annotate(series, (ks[-1], m[-1]), xytext=(6, 0), textcoords="offset points",
                            fontsize=8, color=TEXT, va="center")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Number of training writers k")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Writer diversity vs generalization (baseline arm, mean ± std over 5 seeds)", fontsize=10, color=TEXT)
    fig.tight_layout()
    p = out / "regimes_diversity.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # R5 learning curves
    k = cfg.get("finetune_k", 3)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=150)
    for ax, metric, ylabel in ((axes[0], "unseen_acc", "Unseen-writer accuracy (%)"), (axes[1], "mnist_acc", "MNIST accuracy (%)")):
        for (regime, nw), label, color in (((("R3",), k), f"{k} writers, scratch", "#2a78d6"),
                                           ((("R5",), k), f"{k} writers, MNIST-pretrained", "#eb6834"),
                                           ((("R2",), 1), "1 writer, scratch", "#1baf7a"),
                                           ((("R5",), 1), "1 writer, MNIST-pretrained", "#6250d6")):
            curves = defaultdict(list)
            for r in runs:
                if r["arm"] == "baseline" and r["regime"] in regime and len(r["train_writers"]) == nw and r.get("curve"):
                    for pt in r["curve"]:
                        if pt.get(metric) is not None:
                            curves[pt["epoch"]].append(pt[metric])
            if curves:
                es = sorted(curves)
                ax.plot(es, [100 * np.mean(curves[e]) for e in es], marker="o", ms=4, lw=2, color=color, label=label)
        ax.set_xlabel("Epoch (0 = before fine-tuning)")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("R5 fine-tuning vs training from scratch (baseline arm, mean over seeds and combinations)", fontsize=10, color=TEXT)
    fig.tight_layout()
    p = out / "regimes_finetune_curves.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # RQ1 overview: seen / unseen / MNIST accuracy and WriterBias per regime
    groups = [g for g in ("R1", "R2", "R3 k=3", "R4", "R5 (1 writer)", "R5 (3 writers)") if (g, "baseline") in agg]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150, gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(groups))
    bars = (("seen_acc", "Seen-writer held-out (R1: MNIST test)", "#2a78d6"),
            ("unseen_acc", "Unseen writers (R1: all 5)", "#eb6834"),
            ("mnist_acc", "MNIST test", "#1baf7a"))
    w = 0.26
    for i, (metric, label, color) in enumerate(bars):
        m = [100 * agg[(g, "baseline")][metric][0] if metric in agg[(g, "baseline")] else np.nan for g in groups]
        s = [100 * agg[(g, "baseline")][metric][1] if metric in agg[(g, "baseline")] else 0 for g in groups]
        axes[0].bar(x + (i - 1) * w, m, w - 0.03, yerr=s, capsize=2, color=color, label=label,
                    error_kw={"lw": 0.8, "ecolor": MUTED})
    axes[0].set_xticks(x, groups, fontsize=8)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 105)
    axes[0].legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    axes[0].grid(axis="y", color=GRID, lw=0.8)
    _style(axes[0])
    m = [100 * agg[(g, "baseline")]["writer_bias"][0] for g in groups]
    s = [100 * agg[(g, "baseline")]["writer_bias"][1] for g in groups]
    axes[1].bar(x, m, 0.6, yerr=s, capsize=2, color="#2a78d6", error_kw={"lw": 0.8, "ecolor": MUTED})
    axes[1].set_xticks(x, groups, fontsize=7, rotation=30, ha="right")
    axes[1].set_ylabel("WriterBias, 5 writers (pp)")
    axes[1].grid(axis="y", color=GRID, lw=0.8)
    _style(axes[1])
    fig.suptitle("Regimes R1–R5 (baseline arm, mean ± std over 5 seeds)", fontsize=10, color=TEXT)
    fig.tight_layout()
    p = out / "regimes_overview.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # RQ2 size sweep
    fracs = sorted(cfg.get("size_fractions", [])) + [1.0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=150)
    for ax, metric, ylabel in ((axes[0], "seen_acc", "Seen-writer held-out acc (%)"),
                               (axes[1], "gap_acc", "Gap: train − held-out acc (pp)"),
                               (axes[2], "writer_bias", "WriterBias, 5 writers (pp)")):
        for base, color, suffix, ls in (("R3 k=3", "#2a78d6", "", "-"), ("R4", "#eb6834", "", "-"),
                                        ("R3 k=3", "#2a78d6", " eq", "--"), ("R4", "#eb6834", " eq", "--")):
            xs, mm, ss = [], [], []
            for f in fracs:
                g = base if f == 1.0 else f"{base} @{int(round(100 * f))}%{suffix}"
                st = agg.get((g, "baseline"), {}).get(metric)
                if st:
                    xs.append(100 * f)
                    mm.append(100 * st[0])
                    ss.append(100 * st[1])
            if len(xs) > 1:
                lab = ("R3, 3 writers" if base == "R3 k=3" else "R4, all 5 writers") + (", equal steps" if suffix else ", 5 epochs")
                ax.errorbar(xs, mm, yerr=ss, marker="s" if suffix else "o", ms=5, lw=2, ls=ls, capsize=3, color=color, label=lab)
        ax.set_xscale("log")
        ax.set_xticks([10, 25, 50, 100], ["10%", "25%", "50%", "100%"])
        ax.minorticks_off()
        ax.set_xlabel("Share of each writer's training split")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color=GRID, lw=0.8)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Training-set size sweep (baseline arm, mean ± std over 5 seeds)", fontsize=10, color=TEXT)
    fig.tight_layout()
    p = out / "regimes_size_sweep.png"
    fig.savefig(p)
    plt.close(fig)
    paths.append(p)

    # R5 sweep tradeoff
    sw = cfg.get("finetune_sweep")
    if sw:
        k = cfg.get("finetune_k", 3)
        fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=150)
        for (who, scratch), color in zip((("1 writer", "R2"), (f"{k} writers", f"R3 k={k}")), ("#2a78d6", "#eb6834")):
            first = True
            for lr_s in sw["lr_scales"]:
                for ep in sw["epochs"]:
                    a_ = agg.get((r5_group(cfg, who, lr_s, ep), "baseline"), {})
                    sc = agg.get((scratch, "baseline"), {})
                    if "forgetting" not in a_ or "unseen_acc" not in a_:
                        continue
                    gain = 100 * (np.array(a_["unseen_acc"][2]) - np.array(sc["unseen_acc"][2]))
                    fg = 100 * np.array(a_["forgetting"][2])
                    ax.errorbar(fg.mean(), gain.mean(), xerr=fg.std(ddof=1), yerr=gain.std(ddof=1), fmt="o" if ep == 5 else "s",
                                ms=6, color=color, capsize=2, lw=1, label=f"{who} (circle: 5 epochs, square: 15)" if first else None)
                    first = False
                    ax.annotate(f"lr×{lr_s:g}", (fg.mean(), gain.mean()), xytext=(5, 4), textcoords="offset points", fontsize=7, color=TEXT)
        ax.set_xlabel("MNIST forgetting vs R1 (pp)")
        ax.set_ylabel("Unseen-writer gain vs scratch (pp)")
        ax.grid(color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=7)
        _style(ax)
        ax.set_title("R5 fine-tuning: forgetting vs gain", fontsize=10, color=TEXT)
        fig.tight_layout()
        p = out / "regimes_finetune_tradeoff.png"
        fig.savefig(p)
        plt.close(fig)
        paths.append(p)
    return paths


def build(cfg, with_figures=True):
    runs = load_runs(cfg["results_dir"])
    if not runs:
        raise SystemExit("no regime runs yet")
    agg = aggregate(runs)
    out = Path(cfg["results_dir"])
    serial = {f"{g}|{a}": {m: {"mean": v[0], "std": v[1], "per_seed": v[2]} for m, v in d.items()} for (g, a), d in agg.items()}
    (out / "summary.json").write_text(json.dumps(serial, indent=2), encoding="utf-8")
    (out / "REGIMES.md").write_text(markdown(cfg, runs, agg), encoding="utf-8")
    figs = []
    if with_figures:
        fig_dir = out / "figures"
        fig_dir.mkdir(exist_ok=True)
        figs = figures(cfg, agg, runs, fig_dir)
    print(f"[regimes report] {len(runs)} runs -> {out.as_posix()}/REGIMES.md, {len(figs)} figures")
    return agg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/regimes.json")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    build(load_config(a.config), not a.no_figures)


if __name__ == "__main__":
    main()
