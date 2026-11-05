"""The five training regimes from the team's project proposal.

R1  train on MNIST; test on MNIST and on every writer.
R2  train on one writer; test on that writer's held-out split, the other
    writers, and MNIST.
R3  train on k mixed writers (k = 2, 3, 4; every combination); test on seen and
    unseen writers and MNIST. A size-matched variant caps the total training set
    at one writer's size so writer diversity is separated from data size.
R4  train on all 5 writers; test on each writer's split and MNIST.
R3f/R4f  size sweep (RQ2): R3 (k = 3) and R4 trained on 10/25/50% of every
    training writer's split (100% is plain R3/R4).
R5  start from the R1 model (same seed) and fine-tune on R2 or R3 (k = 3) data at
    a 10x lower learning rate; also measures forgetting on MNIST.

Test sets are the same in every regime: each writer's held-out split (the team's
t10k file, 286 samples per writer) and the 10,000-image MNIST test set. Training
on a writer always uses only that writer's train split.

An "arm" is a stylus preprocessing pipeline (see transforms.apply_preprocess). It
is applied to every stylus image, in training and in testing; MNIST images are
never modified.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from digitstudy import data as D
from digitstudy.transforms import apply_preprocess


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg["_config_path"] = Path(path).as_posix()
    cfg.setdefault("workers", 1)
    return cfg


@dataclass(frozen=True)
class Job:
    regime: str  # R1 | R2 | R3 | R3s (size-matched) | R4 | R5
    seed: int
    train_writers: tuple[str, ...]  # empty for R1 (MNIST)
    train_arm: str  # preprocessing for stylus training data ("-" for R1)
    eval_arms: tuple[str, ...]
    per_writer_cap: int | None = None
    finetune: bool = False
    fraction: float | None = None  # size sweep: keep this share of each writer's train split

    def key(self, eval_arm: str) -> str:
        tw = "+".join(self.train_writers) or "mnist"
        cap = self.per_writer_cap or "full"
        frac = f"|frac={self.fraction}" if self.fraction else ""
        return f"{self.regime}|{eval_arm}|{tw}|cap={cap}{frac}|seed={self.seed}"


# ------------------------------------------------------------------ data

_CACHE: dict[str, Any] = {}


def _writers(cfg) -> dict[str, D.WriterData]:
    if "writers" not in _CACHE:
        ws = D.load_stylus()
        cap = cfg.get("max_per_writer")  # smoke test only
        if cap:
            for w in ws.values():
                tr = np.where(~w.team_test_mask)[0][: cap]
                te = np.where(w.team_test_mask)[0][: max(cap // 6, 10)]
                keep = np.concatenate([tr, te])
                w.images, w.labels, w.team_test_mask = w.images[keep], w.labels[keep], w.team_test_mask[keep]
        _CACHE["writers"] = ws
    return _CACHE["writers"]


def _arm_writers(cfg, arm: str) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """writer -> (x_train, y_train, x_test, y_test), preprocessed with `arm`."""
    key = f"arm:{arm}"
    if key not in _CACHE:
        spec = cfg["arms"][arm]
        out = {}
        for name, w in _writers(cfg).items():
            m = w.team_test_mask
            x = apply_preprocess(w.images, spec)
            out[name] = (x[~m], w.labels[~m], x[m], w.labels[m])
        _CACHE[key] = out
    return _CACHE[key]


def _mnist(cfg, split: str):
    key = f"mnist:{split}"
    if key not in _CACHE:
        x, y = D.load_mnist(split)
        n = cfg.get("mnist_train_subset") if split == "train" else cfg.get("mnist_test_subset")
        if n:
            idx = np.random.default_rng(7 if split == "train" else 8).permutation(len(y))[:n]
            x, y = x[idx], y[idx]
        _CACHE[key] = (x, y)
    return _CACHE[key]


# ------------------------------------------------------------------ jobs


def build_jobs(cfg) -> tuple[list[Job], list[Job]]:
    """Return (phase1, phase2). Phase 1 (R1) must finish before R5 can start."""
    writers = D.list_writers()
    seeds = cfg["seeds"]
    arms = list(cfg["arms"])
    sub = cfg.get("size_matched_arms", arms)
    r5_arms = cfg.get("finetune_arms", arms)
    p1 = [Job("R1", s, (), "-", tuple(arms)) for s in seeds]
    p2: list[Job] = []
    one_writer_cap = cfg.get("size_matched_total") or min(int((~w.team_test_mask).sum()) for w in _writers(cfg).values())
    for s in seeds:
        for arm in arms:
            for w in writers:
                p2.append(Job("R2", s, (w,), arm, (arm,)))
            for k in cfg.get("mixed_k", [2, 3, 4]):
                for combo in itertools.combinations(writers, k):
                    p2.append(Job("R3", s, combo, arm, (arm,)))
            p2.append(Job("R4", s, tuple(writers), arm, (arm,)))
        for arm in sub:
            for k in cfg.get("mixed_k", [2, 3, 4]):
                for combo in itertools.combinations(writers, k):
                    p2.append(Job("R3s", s, combo, arm, (arm,), per_writer_cap=one_writer_cap // k))
        for frac in cfg.get("size_fractions", []):
            for arm in cfg.get("size_sweep_arms", ["baseline"]):
                p2.append(Job("R4f", s, tuple(writers), arm, (arm,), fraction=frac))
                for combo in itertools.combinations(writers, 3):
                    p2.append(Job("R3f", s, combo, arm, (arm,), fraction=frac))
        for arm in r5_arms:
            for w in writers:
                p2.append(Job("R5", s, (w,), arm, (arm,), finetune=True))
            for combo in itertools.combinations(writers, cfg.get("finetune_k", 3)):
                p2.append(Job("R5", s, combo, arm, (arm,), finetune=True))
    return p1, p2


# ------------------------------------------------------------------ train / eval


def _evaluate(model, x: np.ndarray, y: np.ndarray, batch_size: int = 1024) -> dict[str, Any]:
    import torch

    model.eval()
    loss_sum, preds = 0.0, []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(x[i : i + batch_size], dtype=np.float32)).unsqueeze(1)
            yb = torch.from_numpy(y[i : i + batch_size])
            logits = model(xb)
            loss_sum += torch.nn.functional.cross_entropy(logits, yb, reduction="sum").item()
            preds.append(logits.argmax(1).numpy())
    p = np.concatenate(preds)
    cm = np.zeros((10, 10), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return {"acc": float((p == y).mean()), "loss": loss_sum / len(y), "n": int(len(y)), "confusion": cm.tolist()}


def _model_path(cfg, seed: int) -> Path:
    return Path(cfg["results_dir"]) / "models" / f"r1_seed{seed}.pt"


def run_job(cfg, job: Job) -> list[dict[str, Any]]:
    import torch

    from digitstudy.torch_model import CNN, set_seed

    if cfg.get("threads_per_worker"):
        torch.set_num_threads(int(cfg["threads_per_worker"]))
    tcfg = cfg["training"]
    started = time.time()

    # --- training data
    if job.regime == "R1":
        x_tr, y_tr = _mnist(cfg, "train")
    else:
        aw = _arm_writers(cfg, job.train_arm)
        rng = np.random.default_rng(job.seed + 1000)
        xs, ys = [], []
        for w in job.train_writers:
            xw, yw = aw[w][0], aw[w][1]
            if job.per_writer_cap:
                idx = np.sort(rng.permutation(len(yw))[: job.per_writer_cap])
                xw, yw = xw[idx], yw[idx]
            elif job.fraction:
                idx = np.sort(rng.permutation(len(yw))[: max(10, int(round(job.fraction * len(yw))))])
                xw, yw = xw[idx], yw[idx]
            xs.append(xw)
            ys.append(yw)
        x_tr, y_tr = np.concatenate(xs), np.concatenate(ys)

    # --- model
    set_seed(job.seed)
    model = CNN(**cfg["model"])
    lr = float(tcfg["learning_rate"])
    if job.finetune:
        model.load_state_dict(torch.load(_model_path(cfg, job.seed), weights_only=True))
        lr *= float(cfg.get("finetune_lr_scale", 0.1))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    x_t = torch.from_numpy(np.ascontiguousarray(x_tr, dtype=np.float32)).unsqueeze(1)
    y_t = torch.from_numpy(y_tr)
    rng = np.random.default_rng(job.seed + 1)

    # per-epoch curve on the job's own train arm
    curve_arm = job.eval_arms[0]
    seen = set(job.train_writers)

    def curve_point(epoch: int, train_loss: float) -> dict[str, Any]:
        aw = _arm_writers(cfg, curve_arm)
        accs = {w: _evaluate(model, v[2], v[3])["acc"] for w, v in aw.items()}
        s = [a for w, a in accs.items() if w in seen]
        u = [a for w, a in accs.items() if w not in seen]
        xm, ym = _mnist(cfg, "test")
        return {"epoch": epoch, "train_loss": train_loss, "seen_acc": float(np.mean(s)) if s else None,
                "unseen_acc": float(np.mean(u)) if u else None, "mnist_acc": _evaluate(model, xm, ym)["acc"]}

    curve = [curve_point(0, float("nan"))] if job.finetune else []
    bs = int(tcfg["batch_size"])
    for epoch in range(1, int(tcfg["epochs"]) + 1):
        model.train()
        order = torch.from_numpy(rng.permutation(len(y_tr)))
        total = 0.0
        for i in range(0, len(y_tr), bs):
            idx = order[i : i + bs]
            opt.zero_grad()
            loss = loss_fn(model(x_t[idx]), y_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if job.regime != "R1":
            curve.append(curve_point(epoch, total / len(y_tr)))

    if job.regime == "R1":
        path = _model_path(cfg, job.seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    train_eval = _evaluate(model, x_tr, y_tr)
    seconds = time.time() - started
    records = []
    xm, ym = _mnist(cfg, "test")
    mnist_eval = _evaluate(model, xm, ym)
    for arm in job.eval_arms:
        tests = {"mnist": mnist_eval}
        for w, v in _arm_writers(cfg, arm).items():
            tests[w] = _evaluate(model, v[2], v[3])
        records.append({
            "run_key": job.key(arm), "regime": job.regime, "arm": arm, "seed": job.seed,
            "train_writers": list(job.train_writers), "per_writer_cap": job.per_writer_cap,
            "fraction": job.fraction,
            "finetune": job.finetune, "n_train": int(len(y_tr)),
            "train_acc": train_eval["acc"], "train_loss": train_eval["loss"],
            "tests": tests, "curve": curve, "seconds": seconds,
        })
    return records


def _entry(args):
    return run_job(*args)


def load_runs(results_dir) -> list[dict[str, Any]]:
    p = Path(results_dir) / "runs.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_regimes(cfg) -> None:
    out = Path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    p1, p2 = build_jobs(cfg)
    workers = max(1, int(cfg["workers"]))
    started = time.time()
    for phase, jobs in (("R1", p1), ("R2-R5", p2)):
        done = {r["run_key"] for r in load_runs(out)}
        if phase == "R2-R5":
            # R5 needs the saved R1 weights; rerun R1 for a seed whose weights are missing
            missing = [s for s in cfg["seeds"] if not _model_path(cfg, s).exists()]
            if missing:
                raise SystemExit(f"Missing R1 weights for seeds {missing}; delete R1 rows from runs.jsonl and rerun.")
        todo = [j for j in jobs if not all(j.key(a) in done for a in j.eval_arms)]
        print(f"[regimes] phase {phase}: {len(todo)} jobs to run", flush=True)

        def write(records, i):
            with runs_path.open("a", encoding="utf-8") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
            r = records[0]
            print(f"[{phase} {i}/{len(todo)}] {r['run_key']} mnist={r['tests']['mnist']['acc']:.4f} "
                  f"({r['seconds']:.1f}s, elapsed {time.time() - started:.0f}s)", flush=True)

        if workers == 1:
            for i, j in enumerate(todo, 1):
                write(run_job(cfg, j), i)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futs = [pool.submit(_entry, (cfg, j)) for j in todo]
                for i, f in enumerate(as_completed(futs), 1):
                    write(f.result(), i)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run the five training regimes.")
    ap.add_argument("--config", default="configs/regimes.json")
    args = ap.parse_args(argv)
    run_regimes(load_config(args.config))


if __name__ == "__main__":
    main()
