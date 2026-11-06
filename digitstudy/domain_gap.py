"""MNIST <-> stylus domain gap: measurements, one-factor ablation, reverse degradation and fixes.

Two transfer directions, each with its own validation set used for recipe selection:

  s2m  train on all 10,000 stylus digits; validate on 10,000 held-out MNIST *train* images;
       test on the 10,000 MNIST test images.
  m2s  train on the other 50,000 MNIST train images; validate on the stylus writers' train splits
       (8,566); test on their held-out splits (1,434), averaged over the 5 writers.

A recipe has deterministic preprocessing (pre_train on the source, pre_test on the target,
pre_both on both) and train-time batch augmentations (see augment_batch).

Phases (configs/domain_gap.json):
  select   candidate recipes, selection seeds, validation accuracy only (test never computed).
           The best recipe per direction is written to the config as the pre-registered headline.
  final    every recipe, seeds 0-4, validation and test accuracy, confusion matrices.
  reverse  the R1 models (MNIST-trained, results/regimes/models) on MNIST test images degraded
           towards stylus style, one factor at a time and all together.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from digitstudy import data as D
from digitstudy import transforms as T

CACHE_DIR = Path("results/domain_gap/cache")


def load_config(path="configs/domain_gap.json") -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg["_config_path"] = Path(path).as_posix()
    return cfg


# ------------------------------------------------------------------ data


_MEM: dict[str, Any] = {}


def _raw(cfg) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if "raw" in _MEM:
        return _MEM["raw"]
    ws = D.load_stylus()
    names = sorted(ws)
    st_all = (np.concatenate([ws[n].images for n in names]), np.concatenate([ws[n].labels for n in names]))
    st_tr = (np.concatenate([ws[n].images[~ws[n].team_test_mask] for n in names]),
             np.concatenate([ws[n].labels[~ws[n].team_test_mask] for n in names]))
    st_te_w = np.concatenate([np.full(int(ws[n].team_test_mask.sum()), i) for i, n in enumerate(names)])
    st_te = (np.concatenate([ws[n].images[ws[n].team_test_mask] for n in names]),
             np.concatenate([ws[n].labels[ws[n].team_test_mask] for n in names]))
    mx, my = D.load_mnist("train")
    perm = np.random.default_rng(11).permutation(len(my))
    nv = int(cfg.get("mnist_val_size", 10000))
    val_idx, tr_idx = np.sort(perm[:nv]), np.sort(perm[nv:])
    mt = D.load_mnist("test")
    out = {"stylus_all": st_all, "stylus_train": st_tr, "stylus_test": st_te,
           "mnist_val": (mx[val_idx], my[val_idx]), "mnist_rest": (mx[tr_idx], my[tr_idx]), "mnist_test": mt}
    small = cfg.get("smoke_n")
    if small:
        rng = np.random.default_rng(0)
        for k, (x, y) in list(out.items()):
            idx = np.sort(rng.permutation(len(y))[:small])
            out[k] = (x[idx], y[idx])
            if k == "stylus_test":
                st_te_w = st_te_w[idx]
    _MEM["raw"] = out
    _MEM["stylus_test_writer"] = st_te_w
    _MEM["writer_names"] = names
    return out


def _prep(cfg, name: str, spec: dict | None) -> np.ndarray:
    """Preprocessed images for dataset `name`, cached on disk by (dataset, spec)."""
    x, _ = _raw(cfg)[name]
    if not spec:
        return x
    tag = hashlib.sha1(json.dumps([name, spec, cfg.get("smoke_n")], sort_keys=True).encode()).hexdigest()[:16]
    key = f"{name}:{tag}"
    if key in _MEM:
        return _MEM[key]
    path = CACHE_DIR / f"{tag}.npy"
    if path.exists():
        arr = np.load(path)
    else:
        arr = T.apply_preprocess(x, spec)
        if not cfg.get("smoke_n"):
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{np.random.randint(1e9)}.tmp.npy")
            np.save(tmp, arr)
            tmp.replace(path)
    _MEM[key] = arr
    return arr


def _merge(*specs):
    out = {}
    for s in specs:
        out.update(s or {})
    return out


DATASETS = {"s2m": ("stylus_all", "mnist_val", "mnist_test"), "m2s": ("mnist_rest", "stylus_train", "stylus_test")}


def datasets(cfg, direction: str, recipe: dict, need_test: bool):
    src, val, test = DATASETS[direction]
    raw = _raw(cfg)
    both = recipe.get("pre_both")
    xs = _prep(cfg, src, _merge(both, recipe.get("pre_train")))
    tgt_spec = _merge(both, recipe.get("pre_test"))
    xv = _prep(cfg, val, tgt_spec)
    xt = _prep(cfg, test, tgt_spec) if need_test else None
    return (xs, raw[src][1]), (xv, raw[val][1]), ((xt, raw[test][1]) if need_test else None)


# ------------------------------------------------------------------ augmentation


def _gauss_kernel(sigma: float, radius: int):
    import torch

    t = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _sep_blur(x, sigma: float):
    import torch.nn.functional as F

    r = max(1, int(3 * sigma))
    k = _gauss_kernel(sigma, r).to(x.dtype)
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0)), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r)), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def augment_batch(x, augs: list[str], g, params: dict[str, Any] | None = None):
    """Random per-sample augmentation of a (B, 1, 28, 28) tensor in [0, 1].

    rand_dilate  with p = 0.5, a 3x3 grayscale dilation (stroke thickening, k in {0, 1})
    rand_erode   with p = 0.5, a 2x2 grayscale erosion (stroke thinning)
    noise        additive Gaussian noise, sigma 0.1, clipped
    blur         with p = 0.5, Gaussian blur, sigma 0.7, peak restored
    contrast     multiply by U(0.4, 1.0)
    hard         with p = 0.5, binarize at 0.4 (removes anti-aliasing)
    slant        horizontal shear U(-0.3, 0.3) (about +-17 deg; MNIST's slant spread)
    affine       rotation +-12 deg, scale 0.9-1.1, shear +-0.25, shift +-2 px
    elastic      Simard et al. elastic distortion, alpha 34 px, sigma 4 px
    """
    import torch
    import torch.nn.functional as F

    p = params or {}
    B = x.shape[0]

    def coin(prob=0.5):
        return (torch.rand(B, generator=g) < prob).view(B, 1, 1, 1).to(x.dtype)

    for a in augs:
        if a == "rand_dilate":
            x = torch.where(coin() > 0, F.max_pool2d(x, 3, stride=1, padding=1), x)
        elif a == "rand_erode":
            er = -F.max_pool2d(-F.pad(x, (0, 1, 0, 1)), 2, stride=1)
            x = torch.where(coin() > 0, er, x)
        elif a == "noise":
            x = (x + p.get("noise_sigma", 0.1) * torch.randn(x.shape, generator=g)).clamp(0, 1)
        elif a == "blur":
            b = _sep_blur(x, p.get("blur_sigma", 0.7))
            peak = x.amax((2, 3), keepdim=True) / b.amax((2, 3), keepdim=True).clamp_min(1e-6)
            x = torch.where(coin() > 0, (b * peak).clamp(0, 1), x)
        elif a == "contrast":
            x = x * (0.4 + 0.6 * torch.rand(B, 1, 1, 1, generator=g))
        elif a == "hard":
            x = torch.where(coin() > 0, (x > 0.4).to(x.dtype), x)
        elif a == "affine":
            ang = (torch.rand(B, generator=g) * 2 - 1) * np.radians(12)
            sc = 0.9 + 0.2 * torch.rand(B, generator=g)
            sh = (torch.rand(B, generator=g) * 2 - 1) * 0.25
            tx, ty = [(torch.rand(B, generator=g) * 2 - 1) * (2 / 14) for _ in range(2)]
            cos, sin = torch.cos(ang) / sc, torch.sin(ang) / sc
            theta = torch.stack([torch.stack([cos, -sin + sh * cos, tx], 1), torch.stack([sin, cos + sh * sin, ty], 1)], 1)
            grid = F.affine_grid(theta, x.shape, align_corners=False)
            x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        elif a == "slant":
            sh = (torch.rand(B, generator=g) * 2 - 1) * p.get("slant_max", 0.3)
            z, o = torch.zeros(B), torch.ones(B)
            theta = torch.stack([torch.stack([o, sh, z], 1), torch.stack([z, o, z], 1)], 1)
            x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), mode="bilinear",
                              padding_mode="zeros", align_corners=False)
        elif a == "elastic":
            alpha, sigma = 34.0, 4.0
            disp = torch.rand(B, 2, 28, 28, generator=g) * 2 - 1
            disp = _sep_blur(disp, sigma) * alpha * (2 / 28)
            base = F.affine_grid(torch.eye(2, 3).unsqueeze(0).repeat(B, 1, 1), x.shape, align_corners=False)
            x = F.grid_sample(x, base + disp.permute(0, 2, 3, 1), mode="bilinear", padding_mode="zeros", align_corners=False)
        else:
            raise ValueError(f"unknown augmentation {a}")
    return x


# ------------------------------------------------------------------ train / eval


def _device():
    """Training device: CPU by default; DIGITSTUDY_DEVICE=dml uses DirectML (needs torch-directml)."""
    import os

    if os.environ.get("DIGITSTUDY_DEVICE", "cpu") == "dml":
        import torch_directml

        return torch_directml.device(), "dml"
    return None, "cpu"


def _evaluate(model, x, y, groups=None, dev=None):
    import torch

    model.eval()
    preds, loss = [], 0.0
    with torch.no_grad():
        for i in range(0, len(y), 2048):
            xb = torch.from_numpy(np.ascontiguousarray(x[i:i + 2048], dtype=np.float32)).unsqueeze(1)
            lo = model(xb if dev is None else xb.to(dev)).cpu()
            loss += torch.nn.functional.cross_entropy(lo, torch.from_numpy(y[i:i + 2048]), reduction="sum").item()
            preds.append(lo.argmax(1).numpy())
    p = np.concatenate(preds)
    cm = np.zeros((10, 10), np.int64)
    np.add.at(cm, (y, p), 1)
    out = {"acc": float((p == y).mean()), "loss": loss / len(y), "n": int(len(y)), "confusion": cm.tolist()}
    if groups is not None:
        per = [float((p[groups == k] == y[groups == k]).mean()) for k in np.unique(groups)]
        out["per_writer"] = per
        out["macro_writer_acc"] = float(np.mean(per))
    return out, p


def run_one(cfg, phase: str, direction: str, recipe_name: str, seed: int) -> dict[str, Any]:
    import torch

    from digitstudy.torch_model import CNN, set_seed

    torch.set_num_threads(int(cfg.get("threads_per_worker", 4)))
    dev, dev_name = _device()
    recipe = cfg["recipes"][recipe_name]
    need_test = phase == "final"
    (xs, ys), (xv, yv), test = datasets(cfg, direction, recipe, need_test)
    t0 = time.time()
    set_seed(seed)
    model = CNN(**cfg["model"])
    if dev is not None:
        model = model.to(dev)
    tc = cfg["training"]
    opt = torch.optim.Adam(model.parameters(), lr=float(tc["learning_rate"]))
    lossf = torch.nn.CrossEntropyLoss()
    x_t = torch.from_numpy(np.ascontiguousarray(xs, dtype=np.float32)).unsqueeze(1)
    y_t = torch.from_numpy(ys)
    g = torch.Generator().manual_seed(seed + 12345)
    rng = np.random.default_rng(seed + 1)
    augs = recipe.get("aug", [])
    bs = int(tc["batch_size"])
    for _ in range(int(tc["epochs"])):
        model.train()
        order = torch.from_numpy(rng.permutation(len(ys)))
        for i in range(0, len(ys), bs):
            idx = order[i:i + bs]
            xb = x_t[idx]
            if augs:
                xb = augment_batch(xb, augs, g, recipe.get("aug_params"))
            yb = y_t[idx]
            if dev is not None:
                xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    groups_val = None
    rec = {"run_key": f"{phase}|{direction}|{recipe_name}|seed={seed}", "phase": phase, "direction": direction,
           "recipe": recipe_name, "seed": seed, "n_train": int(len(ys)), "device": dev_name}
    rec["val"], _ = _evaluate(model, xv, yv, groups_val, dev)
    if need_test:
        groups = _MEM["stylus_test_writer"] if direction == "m2s" else None
        rec["test"], pred = _evaluate(model, test[0], test[1], groups, dev)
        rec["test"]["score"] = rec["test"].get("macro_writer_acc", rec["test"]["acc"])
        if seed == cfg["seeds"][0]:
            pdir = Path(cfg["results_dir"]) / "preds"
            pdir.mkdir(parents=True, exist_ok=True)
            np.save(pdir / f"{direction}__{recipe_name}.npy", pred.astype(np.uint8))
    rec["seconds"] = time.time() - t0
    return rec


def _entry(a):
    return run_one(*a)


def load_runs(cfg) -> list[dict[str, Any]]:
    p = Path(cfg["results_dir"]) / "runs.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _run_jobs(cfg, jobs):
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    done = {r["run_key"] for r in load_runs(cfg)}
    todo = [j for j in jobs if f"{j[0]}|{j[1]}|{j[2]}|seed={j[3]}" not in done]
    # warm the disk cache in the parent so workers do not all preprocess the same arrays
    for ph, d, r, _ in {(j[0], j[1], j[2], 0) for j in todo}:
        datasets(cfg, d, cfg["recipes"][r], ph == "final")
    print(f"[domain-gap] {len(todo)} runs to do", flush=True)
    t0 = time.time()
    import os

    w = int(os.environ.get("DIGITSTUDY_WORKERS") or cfg.get("workers", 1))

    def write(rec, i):
        with (out / "runs.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        tst = f" test={rec['test']['score']:.4f}" if "test" in rec else ""
        print(f"[{i}/{len(todo)}] {rec['run_key']} val={rec['val']['acc']:.4f}{tst} "
              f"({rec['seconds']:.0f}s, elapsed {time.time() - t0:.0f}s)", flush=True)

    if w <= 1:
        for i, j in enumerate(todo, 1):
            write(run_one(cfg, *j), i)
    else:
        with ProcessPoolExecutor(max_workers=w) as pool:
            futs = [pool.submit(_entry, (cfg, *j)) for j in todo]
            for i, f in enumerate(as_completed(futs), 1):
                write(f.result(), i)


def select(cfg):
    jobs = [("select", d, r, s) for d, names in cfg["candidates"].items() for r in names for s in cfg["selection_seeds"]]
    _run_jobs(cfg, jobs)
    runs = [r for r in load_runs(cfg) if r["phase"] == "select"]
    cfg_raw = json.loads(Path(cfg["_config_path"]).read_text(encoding="utf-8"))
    head = cfg_raw.setdefault("preregistration", {})
    table = {}
    for d, names in cfg["candidates"].items():
        scores = {n: float(np.mean([r["val"]["acc"] for r in runs if r["direction"] == d and r["recipe"] == n]))
                  for n in names}
        table[d] = dict(sorted(scores.items(), key=lambda kv: -kv[1]))
        best = max(scores, key=scores.get)
        if head.get(f"headline_{d}"):
            print(f"[select] {d}: headline already fixed as {head[f'headline_{d}']} (best on val now: {best})")
            continue
        head[f"headline_{d}"] = best
        head[f"headline_{d}_note"] = (f"Chosen automatically on {dt.date.today().isoformat()} as the best mean validation accuracy "
                                      f"over selection seeds {cfg['selection_seeds']}, before any final (test) run.")
    head["selection_val_acc"] = table
    out = Path(cfg["results_dir"])
    (out / "preregistered.json").write_text(json.dumps(head, indent=2), encoding="utf-8")
    if not cfg.get("smoke_n"):  # the smoke config stays untouched
        Path(cfg["_config_path"]).write_text(json.dumps(cfg_raw, indent=2), encoding="utf-8")
    for d, t in table.items():
        print(d, json.dumps(t, indent=1))


def _prereg(cfg) -> dict[str, Any]:
    pre = dict(cfg.get("preregistration", {}))
    f = Path(cfg["results_dir"]) / "preregistered.json"
    if cfg.get("smoke_n") and f.exists():
        pre.update(json.loads(f.read_text(encoding="utf-8")))
    return pre


def final(cfg):
    pre = _prereg(cfg)
    for d in cfg["candidates"]:
        if not pre.get(f"headline_{d}"):
            raise SystemExit(f"No pre-registered headline for {d}; run `select` first.")
    jobs = []
    for d, names in cfg["final"].items():
        names = list(dict.fromkeys(names + [pre[f"headline_{d}"]]))
        jobs += [("final", d, r, s) for r in names for s in cfg["seeds"]]
    _run_jobs(cfg, jobs)


# ------------------------------------------------------------------ reverse degradation


def degrade(x: np.ndarray, kind: str, ref: dict[str, np.ndarray], rng) -> np.ndarray:
    """Move MNIST images towards the measured stylus style (one factor)."""
    from digitstudy.gapstats import measure

    if kind == "thin":  # one-pixel strokes, then the stylus intensity profile
        return T.thin(x) * float(np.median(ref["mean_ink"]))
    if kind == "thin_hard":
        return T.thin(x)
    if kind == "hard":
        return T.binarize(x, 0.5)
    if kind == "dim":
        return T.brightness(x, float(np.mean(ref["mean_ink"]) / 0.69))
    if kind == "uncenter":
        m = measure(x)
        k = rng.integers(0, len(ref["com_dx"]), len(x))
        dx = np.round(ref["com_dx"][k] - m["com_dx"]).astype(int)
        dy = np.round(ref["com_dy"][k] - m["com_dy"]).astype(int)
        return T.shift_int(x, dx, dy)
    if kind == "narrow":
        m = measure(x)
        k = rng.integers(0, len(ref["aspect"]), len(x))
        return T.hstretch(x, np.clip(ref["aspect"][k] / np.maximum(m["aspect"], 0.2), 0.4, 1.2))
    if kind == "upright":
        return T.shear_to(x, 0.0)
    raise ValueError(kind)


def reverse(cfg):
    import torch

    from digitstudy.gapstats import measure
    from digitstudy.torch_model import CNN

    raw = _raw(cfg)
    xs = raw["stylus_all"][0]
    ref = measure(xs)
    xt, yt = raw["mnist_test"]
    rcfg = cfg["reverse"]
    out = Path(cfg["results_dir"])
    variants = {}
    rng = np.random.default_rng(5)
    for name, steps in rcfg["variants"].items():
        x = xt
        for s in steps:
            x = degrade(x, s, ref, rng)
        variants[name] = x.astype(np.float32)
    np.savez_compressed(out / "reverse_examples.npz", **{k: v[:20] for k, v in variants.items()}, labels=yt[:20])
    recs = []
    for seed in cfg["seeds"]:
        model = CNN(**cfg["model"])
        model.load_state_dict(torch.load(Path(rcfg["models_dir"]) / f"r1_seed{seed}.pt", weights_only=True))
        for name, x in variants.items():
            ev, _ = _evaluate(model, x, yt)
            recs.append({"phase": "reverse", "variant": name, "seed": seed, "test": ev})
            print(f"[reverse] seed {seed} {name}: {ev['acc']:.4f}", flush=True)
    (out / "reverse.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")


def factors_table(cfg):
    from digitstudy.gapstats import FACTORS, measure

    raw = _raw(cfg)
    mx = np.concatenate([raw["mnist_val"][0], raw["mnist_rest"][0]])
    ms = measure(mx)
    ss = measure(raw["stylus_all"][0])
    rows = {}
    for k, *_ in FACTORS:
        rows[k] = {"mnist_mean": float(ms[k].mean()), "mnist_std": float(ms[k].std(ddof=1)),
                   "stylus_mean": float(ss[k].mean()), "stylus_std": float(ss[k].std(ddof=1))}
    # per-class closed-loop rates for the 9 -> 4 question
    ym = np.concatenate([raw["mnist_val"][1], raw["mnist_rest"][1]])
    ys = raw["stylus_all"][1]
    rows["_holes_by_class"] = {
        "mnist": [float((ms["holes"][ym == c] > 0).mean()) for c in range(10)],
        "stylus": [float((ss["holes"][ys == c] > 0).mean()) for c in range(10)]}
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "factors.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    np.savez_compressed(out / "factor_samples.npz",
                        **{f"mnist_{k}": ms[k][:20000].astype(np.float32) for k, *_ in FACTORS},
                        **{f"stylus_{k}": ss[k].astype(np.float32) for k, *_ in FACTORS})
    print(f"[domain-gap] factors -> {out / 'factors.json'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["factors", "select", "final", "reverse", "all"])
    ap.add_argument("--config", default="configs/domain_gap.json")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    phases = ["factors", "select", "final", "reverse"] if a.phase == "all" else [a.phase]
    for ph in phases:
        if ph == "select":
            select(cfg)
            cfg = load_config(a.config)
        else:
            {"factors": factors_table, "final": final, "reverse": reverse}[ph](cfg)


if __name__ == "__main__":
    main()
