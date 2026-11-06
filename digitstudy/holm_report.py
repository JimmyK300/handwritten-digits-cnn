"""Holm correction across every exploratory test in the project -> results/HOLM.md.

Confirmatory tests (named in a config before their runs) are listed but not corrected:
  * dilation study: 3x3 dilation (dilate_k3_i1), PyTorch, leave-one-writer-out and stylus -> MNIST
  * regimes: size-matched series, k = 4 vs k = 1, unseen-writer accuracy and WriterBias
  * domain gap: the validation-selected headline recipe vs baseline, in each direction
Every other p-value reported in RESULTS.md, REGIMES.md or DOMAIN_GAP.md forms ONE Holm family.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from digitstudy import regime_report as RR
from digitstudy.regimes import load_config as load_regimes_config
from digitstudy.regimes import load_runs as load_regime_runs
from digitstudy.stats import holm, paired


def _study_tests():
    p = Path("results/summary.json")
    if not p.exists():
        return []
    out = []
    for r in json.loads(p.read_text(encoding="utf-8")):
        if r["condition"] == "baseline" or r.get("paired_t_p") is None:
            continue
        conf = r["model"] == "torch" and r["condition"] == "dilate_k3_i1" and r["protocol"] in ("lowo", "custom_to_mnist")
        out.append({"family": "Dilation study", "test": f"{r['model']}, {r['protocol']}: {r['condition']} vs baseline",
                    "effect": f"{r['delta_pp_mean']:+.2f} pp", "p": r["paired_t_p"], "confirmatory": conf})
    return out


def _regime_tests():
    cfg = load_regimes_config("configs/regimes.json")
    runs = load_regime_runs(cfg["results_dir"])
    if not runs:
        return []
    agg = RR.aggregate(runs)
    out = []

    def add(test, a, b, scale=100.0, unit="pp", conf=False, fam="Regimes"):
        if a is None or b is None or len(a) != len(b) or len(a) < 2:
            return
        dm, _, p = paired(a, b, scale)
        out.append({"family": fam, "test": test, "effect": f"{dm:+.3g} {unit}", "p": p, "confirmatory": conf})

    def vals(group, metric, arm="baseline"):
        s = agg.get((group, arm), {}).get(metric)
        return s[2] if s else None

    for series, fmt in (("full data", lambda k: "R2" if k == 1 else f"R3 k={k}"),
                        ("size-matched", lambda k: "R2" if k == 1 else f"R3s k={k}")):
        for metric in ("unseen_acc", "writer_bias", "gap_acc", "cross_gap"):
            conf = series == "size-matched" and metric in ("unseen_acc", "writer_bias")
            add(f"diversity ({series}) k=4 vs k=1: {metric}", vals(fmt(1), metric), vals(fmt(4), metric), conf=conf)
    for suffix, label in (("", "5 epochs"), (" eq", "equal steps")):
        for base in ("R3 k=3", "R4"):
            for metric in ("gap_acc", "gap_loss", "writer_bias", "seen_acc"):
                sc, unit = (1.0, "nats") if metric == "gap_loss" else (100.0, "pp")
                add(f"size sweep ({label}) {base} 100% vs 10%: {metric}", vals(f"{base} @10%{suffix}", metric), vals(base, metric), sc, unit)
    k = cfg.get("finetune_k", 3)
    for scratch, ft in (("R2", "R5 (1 writer)"), (f"R3 k={k}", f"R5 ({k} writers)")):
        for metric in ("unseen_acc", "mnist_acc", "writer_bias"):
            add(f"{ft} vs {scratch}: {metric}", vals(scratch, metric), vals(ft, metric))
    for grp in RR.GROUP_ORDER:
        metric = "mnist_acc" if grp == "R4" else "unseen_acc"
        for arm in cfg["arms"]:
            if arm != "baseline":
                add(f"arm {arm} vs baseline, {grp}: {metric}", vals(grp, metric), vals(grp, metric, arm))
    for grp in ("R2", "R3 k=3"):
        for arm in cfg["arms"]:
            if arm != "baseline":
                add(f"arm {arm} vs baseline, {grp}: mnist_acc", vals(grp, "mnist_acc"), vals(grp, "mnist_acc", arm))
    sw = cfg.get("finetune_sweep")
    if sw:
        for who, scratch in (("1 writer", "R2"), (f"{k} writers", f"R3 k={k}")):
            for lr_s in sw["lr_scales"]:
                for ep in sw["epochs"]:
                    g = RR.r5_group(cfg, who, lr_s, ep)
                    if g != f"R5 ({who})":
                        add(f"R5 sweep {who} lr×{lr_s:g} {ep} ep: unseen gain vs {scratch}", vals(scratch, "unseen_acc"), vals(g, "unseen_acc"))
                    f = vals(g, "forgetting")
                    if f:
                        add(f"R5 sweep {who} lr×{lr_s:g} {ep} ep: forgetting > 0", [0.0] * len(f), f)
    return out


def _domain_tests():
    p = Path("results/domain_gap/summary.json")
    if not p.exists():
        return []
    s = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for r in s["comparisons"]:
        if "p" in r:
            out.append({"family": "Domain gap", "test": f"{r['direction']}: {r['recipe']} vs baseline",
                        "effect": f"{r['delta']:+.2f} pp", "p": r["p"], "confirmatory": bool(r["headline"])})
    rev = s.get("reverse", {})
    if "clean" in rev:
        for name, v in rev.items():
            if name != "clean":
                dm, _, pp = paired(rev["clean"], v)
                out.append({"family": "Domain gap", "test": f"reverse degradation: {name} vs clean (R1 models)",
                            "effect": f"{dm:+.2f} pp", "p": pp, "confirmatory": False})
    return out


def build():
    tests = _study_tests() + _regime_tests() + _domain_tests()
    expl = [t for t in tests if not t["confirmatory"] and t["p"] is not None and not np.isnan(t["p"])]
    for t, a in zip(expl, holm([t["p"] for t in expl])):
        t["p_holm"] = a
    conf = [t for t in tests if t["confirmatory"]]
    n_surv = sum(t["p_holm"] < 0.05 for t in expl)
    n_raw = sum(t["p"] < 0.05 for t in expl)
    L = ["# Holm correction across all exploratory tests", "",
         "Built by `python -m digitstudy.holm_report` from `results/summary.json`, `results/regimes/runs.jsonl` and "
         "`results/domain_gap/summary.json`. All tests are two-sided paired t-tests over 5 seeds.", "",
         f"- Exploratory tests: **{len(expl)}**, one Holm family. Significant at p < 0.05 before correction: {n_raw}; "
         f"after Holm: **{n_surv}**.",
         f"- Confirmatory tests (pre-registered, not corrected): {len(conf)}.", "",
         "## Confirmatory", "", "| Family | Test | Effect | p |", "|---|---|---|---|"]
    for t in conf:
        L.append(f"| {t['family']} | {t['test']} | {t['effect']} | {t['p']:.2g} |")
    L += ["", "## Exploratory (Holm-adjusted)", "", "| Family | Test | Effect | p | p (Holm) | Survives |", "|---|---|---|---|---|---|"]
    for t in expl:
        L.append(f"| {t['family']} | {t['test']} | {t['effect']} | {t['p']:.2g} | {t['p_holm']:.2g} | "
                 f"{'yes' if t['p_holm'] < 0.05 else 'no'} |")
    Path("results/HOLM.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    Path("results/holm.json").write_text(json.dumps(tests, indent=1), encoding="utf-8")
    print(f"[holm] {len(expl)} exploratory tests, {n_surv} survive Holm (of {n_raw} with raw p < 0.05); {len(conf)} confirmatory")
    return tests


if __name__ == "__main__":
    build()
