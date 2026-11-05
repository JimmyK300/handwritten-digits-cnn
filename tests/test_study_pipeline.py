"""End-to-end smoke test: tiny study run + report on the real data files."""

import json
from pathlib import Path

import numpy as np

from digitstudy import data as D
from digitstudy.report import build_report, summarize
from digitstudy.study import build_jobs, load_study_config, run_study

ROOT = Path(__file__).resolve().parents[1]


def test_five_writers_about_2000_samples_each():
    writers = D.load_stylus()
    assert len(writers) == 5
    for w in writers.values():
        assert 1990 <= len(w.labels) <= 2000
        assert w.images.shape[1:] == (28, 28)
        assert set(np.unique(w.labels)) == set(range(10))


def test_lowo_never_trains_on_the_held_out_writer():
    cfg = load_study_config(ROOT / "configs" / "smoke.json")
    from digitstudy.study import _split, _stylus

    writers = _stylus(cfg)
    for fold in D.list_writers():
        x_tr, _, parts = _split(cfg, "lowo", fold)
        assert [p[0] for p in parts] == [fold]
        assert len(x_tr) == sum(len(w.labels) for n, w in writers.items() if n != fold)


def test_smoke_study_and_report(tmp_path):
    cfg = load_study_config(ROOT / "configs" / "smoke.json")
    cfg["results_dir"] = str(tmp_path)
    cfg["numpy_cnn"]["protocols"] = ["lowo"]
    run_study(cfg)
    runs = [json.loads(l) for l in (tmp_path / "runs.jsonl").read_text().splitlines()]
    assert len(runs) == sum(len(j.keys) for j in build_jobs(cfg))
    rows = summarize(runs)
    assert any(r["condition"] == "dilate_k3_i1" and "delta_pp_mean" in r for r in rows)
    build_report(cfg, figures=False)
    assert (tmp_path / "RESULTS.md").exists()
    # resuming does no new work
    run_study(cfg)
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == len(runs)
