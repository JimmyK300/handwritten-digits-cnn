"""Regimes runner: job construction and a tiny end-to-end run."""

from pathlib import Path

from digitstudy import regime_report, regimes

ROOT = Path(__file__).resolve().parents[1]


def test_job_counts_follow_the_design():
    cfg = regimes.load_config(ROOT / "configs" / "regimes.json")
    p1, p2 = regimes.build_jobs(cfg)
    assert len(p1) == 5  # R1, one per seed
    counts = {}
    for j in p2:
        counts[j.regime] = counts.get(j.regime, 0) + 1
    seeds, arms = 5, len(cfg["arms"])
    assert counts["R2"] == seeds * arms * 5
    assert counts["R3"] == seeds * arms * (10 + 10 + 5)  # C(5,2)+C(5,3)+C(5,4)
    assert counts["R4"] == seeds * arms
    assert counts["R3s"] == seeds * len(cfg["size_matched_arms"]) * 25
    assert counts["R5"] == seeds * len(cfg["finetune_arms"]) * (5 + 10)
    assert counts["R4f"] == seeds * len(cfg["size_fractions"])
    assert counts["R3f"] == seeds * len(cfg["size_fractions"]) * 10
    # size-matched jobs always use about one writer's worth of training images
    for j in p2:
        if j.regime == "R3s":
            assert j.per_writer_cap * len(j.train_writers) <= 1712


def test_regimes_smoke(tmp_path):
    cfg = regimes.load_config(ROOT / "configs" / "regimes_smoke.json")
    cfg["results_dir"] = str(tmp_path)
    cfg["size_matched_arms"] = []
    regimes.run_regimes(cfg)
    runs = regimes.load_runs(tmp_path)
    assert {r["regime"] for r in runs} == {"R1", "R2", "R3", "R4", "R5", "R3f", "R4f"}
    for r in runs:
        assert set(r["tests"]) >= {"mnist", "Gia Huy"}
    agg = regime_report.build(cfg, with_figures=False)
    assert ("R4", "baseline") in agg
    assert (tmp_path / "REGIMES.md").exists()
