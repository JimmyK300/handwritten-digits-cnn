"""Controlled study: does morphological dilation improve generalization?

Protocols
---------
sample_split     The team's original split: each writer's own train/t10k files,
                 pooled. The same writers appear in train and test.
lowo             Leave-one-writer-out: train on 4 writers, test on the 5th.
                 Repeated for all 5 writers.
mnist_to_stylus  Train on MNIST, test on all stylus samples (cross-domain).

Conditions
----------
Each condition has a deterministic `preprocess` spec (see transforms.py) and an
optional train-time `augment`. In sample_split and lowo the preprocess is applied
to both train and test images. In mnist_to_stylus the network is trained on raw
MNIST and the preprocess is applied to the stylus test images only.

Every finished run is appended as one JSON line to <results_dir>/runs.jsonl,
keyed by run_key, so an interrupted study resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from digitstudy import data as D
from digitstudy.transforms import apply_preprocess, random_stroke_width

AUGMENTS = {
    None: None,
    "stroke_width": random_stroke_width,
}


def load_study_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    cfg.setdefault("results_dir", "results")
    cfg.setdefault("max_per_writer", None)
    cfg.setdefault("mnist_train_subset", None)
    cfg.setdefault("workers", 1)
    cfg["_config_path"] = Path(path).as_posix()
    return cfg


def run_key(protocol: str, model: str, fold: str, seed: int, train_cond: str, eval_cond: str) -> str:
    return f"{protocol}|{model}|{fold}|seed={seed}|train={train_cond}|eval={eval_cond}"


@dataclass
class Job:
    protocol: str
    model: str  # "torch" | "numpy"
    fold: str  # held-out writer for lowo, "all" otherwise
    seed: int
    train_cond: str
    eval_conds: list[str]
    keys: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ data

_CACHE: dict[str, Any] = {}


def _stylus(cfg) -> dict[str, D.WriterData]:
    if "stylus" not in _CACHE:
        writers = D.load_stylus()
        cap = cfg.get("max_per_writer")
        if cap:
            rng = np.random.default_rng(1234)
            for w in writers.values():
                keep = np.sort(rng.permutation(len(w.labels))[:cap])
                w.images, w.labels, w.team_test_mask = w.images[keep], w.labels[keep], w.team_test_mask[keep]
                # make sure the capped sample still has team-test rows
                if not w.team_test_mask.any():
                    w.team_test_mask[: max(1, cap // 7)] = True
        _CACHE["stylus"] = writers
    return _CACHE["stylus"]


def _mnist(cfg) -> tuple[np.ndarray, np.ndarray]:
    if "mnist" not in _CACHE:
        x, y = D.load_mnist("train")
        n = cfg.get("mnist_train_subset")
        if n:
            idx = np.random.default_rng(4321).permutation(len(y))[:n]
            x, y = x[idx], y[idx]
        _CACHE["mnist"] = (x, y)
    return _CACHE["mnist"]


def _split(cfg, protocol: str, fold: str):
    """Return (x_train, y_train, test_parts) where test_parts = [(writer, x, y)]."""
    writers = _stylus(cfg)
    if protocol == "sample_split":
        xs, ys, parts = [], [], []
        for name, w in writers.items():
            m = w.team_test_mask
            xs.append(w.images[~m])
            ys.append(w.labels[~m])
            parts.append((name, w.images[m], w.labels[m]))
        return np.concatenate(xs), np.concatenate(ys), parts
    if protocol == "lowo":
        train = [w for n, w in writers.items() if n != fold]
        x, y = D.concat(train)
        w = writers[fold]
        return x, y, [(fold, w.images, w.labels)]
    if protocol == "mnist_to_stylus":
        x, y = _mnist(cfg)
        return x, y, [(n, w.images, w.labels) for n, w in writers.items()]
    raise ValueError(protocol)


# ------------------------------------------------------------------ jobs


def build_jobs(cfg) -> list[Job]:
    conds = cfg["conditions"]
    writers = D.list_writers()
    jobs: list[Job] = []

    def add(protocol, model, seeds, cond_names):
        for seed in seeds:
            if protocol == "mnist_to_stylus":
                # one trained net per train-time augment; every preprocess is a test-time eval
                pre_only = [c for c in cond_names if not conds[c].get("augment")]
                aug = [c for c in cond_names if conds[c].get("augment")]
                if pre_only:
                    jobs.append(Job(protocol, model, "all", seed, "baseline", pre_only))
                for c in aug:
                    jobs.append(Job(protocol, model, "all", seed, c, [c]))
            else:
                folds = writers if protocol == "lowo" else ["all"]
                for fold in folds:
                    for c in cond_names:
                        jobs.append(Job(protocol, model, fold, seed, c, [c]))

    for protocol in cfg["protocols"]:
        add(protocol, "torch", cfg["seeds"], list(conds))
    np_cfg = cfg.get("numpy_cnn", {})
    if np_cfg.get("enabled"):
        for protocol in np_cfg["protocols"]:
            add(protocol, "numpy", np_cfg["seeds"], np_cfg["conditions"])

    for job in jobs:
        job.keys = [run_key(job.protocol, job.model, job.fold, job.seed, job.train_cond, e) for e in job.eval_conds]
    return jobs


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def run_job(cfg, job: Job) -> list[dict[str, Any]]:
    threads = int(cfg.get("threads_per_worker", 0))
    import torch

    if threads:
        torch.set_num_threads(threads)
    from digitstudy.numpy_cnn import NumpyCNNConfig, train_numpy_cnn
    from digitstudy.torch_model import predict_torch, train_torch

    conds = cfg["conditions"]
    train_spec = conds[job.train_cond]
    x_tr, y_tr, parts = _split(cfg, job.protocol, job.fold)
    if job.protocol != "mnist_to_stylus":
        x_tr = apply_preprocess(x_tr, train_spec.get("preprocess"))
    augment = AUGMENTS[train_spec.get("augment")]

    started = time.time()
    if job.model == "torch":
        model, info = train_torch(x_tr, y_tr, cfg["model"], cfg["training"], job.seed, augment)
        predict = lambda x: predict_torch(model, x)  # noqa: E731
    else:
        np_cfg = cfg["numpy_cnn"]
        if job.protocol == "mnist_to_stylus" and np_cfg.get("mnist_train_subset"):
            idx = np.random.default_rng(99).permutation(len(y_tr))[: np_cfg["mnist_train_subset"]]
            x_tr, y_tr = x_tr[idx], y_tr[idx]
        nmodel, hist = train_numpy_cnn(
            x_tr, y_tr, epochs=np_cfg["epochs"], batch_size=np_cfg["batch_size"], lr=np_cfg["learning_rate"],
            seed=job.seed, augment=augment, config=NumpyCNNConfig(**np_cfg.get("model", {})), log=None,
        )
        info = {"train_loss_history": hist, "num_params": nmodel.num_params()}
        predict = nmodel.predict
    info["train_seconds"] = time.time() - started

    records = []
    for eval_cond, key in zip(job.eval_conds, job.keys):
        pre = conds[eval_cond].get("preprocess")
        y_all, p_all, per_writer = [], [], {}
        for name, x_te, y_te in parts:
            pred = predict(apply_preprocess(x_te, pre))
            per_writer[name] = float((pred == y_te).mean())
            y_all.append(y_te)
            p_all.append(pred)
        y_all, p_all = np.concatenate(y_all), np.concatenate(p_all)
        records.append(
            {
                "run_key": key,
                "protocol": job.protocol,
                "model": job.model,
                "fold": job.fold,
                "seed": job.seed,
                "train_condition": job.train_cond,
                "condition": eval_cond,
                "accuracy": float((p_all == y_all).mean()),
                "n_test": int(len(y_all)),
                "n_train": int(len(y_tr)),
                "per_writer_accuracy": per_writer,
                "confusion": confusion(y_all, p_all).tolist(),
                "train_seconds": info["train_seconds"],
                "train_loss_history": info["train_loss_history"],
                "num_params": int(info["num_params"]),
            }
        )
    return records


def _run_job_entry(args):
    cfg, job = args
    return run_job(cfg, job)


def load_runs(results_dir: str | Path) -> list[dict[str, Any]]:
    path = Path(results_dir) / "runs.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_study(cfg) -> Path:
    out_dir = Path(cfg["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    done = {r["run_key"] for r in load_runs(out_dir)}
    jobs = [j for j in build_jobs(cfg) if not all(k in done for k in j.keys)]
    total = len(jobs)
    print(f"[study] {total} jobs to run ({len(done)} runs already in {runs_path.as_posix()})")
    if not jobs:
        return runs_path

    started = time.time()
    workers = max(1, int(cfg.get("workers", 1)))

    def write(records):
        with runs_path.open("a", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")

    def report(i, records):
        r0 = records[0]
        accs = ", ".join(f"{r['condition']}={r['accuracy']:.4f}" for r in records[:3])
        more = " ..." if len(records) > 3 else ""
        print(f"[{i}/{total}] {r0['protocol']}/{r0['model']}/{r0['fold']}/seed{r0['seed']}: {accs}{more} "
              f"({r0['train_seconds']:.1f}s, elapsed {time.time() - started:.0f}s)", flush=True)

    if workers == 1:
        for i, job in enumerate(jobs, 1):
            records = run_job(cfg, job)
            write(records)
            report(i, records)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_job_entry, (cfg, job)) for job in jobs]
            for i, fut in enumerate(as_completed(futures), 1):
                records = fut.result()
                write(records)
                report(i, records)
    return runs_path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Run the dilation study.")
    parser.add_argument("--config", default="configs/study.json")
    args = parser.parse_args(argv)
    run_study(load_study_config(args.config))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
