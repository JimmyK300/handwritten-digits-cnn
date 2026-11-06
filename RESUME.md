# Resuming the unfinished runs

Both runners append each finished run to a `runs.jsonl` and skip finished runs on restart. If a job is interrupted, rerun the same command; only the runs that were in progress are lost.

## Low-RAM mode (about 3.5 GB in total)

`configs/regimes.json` is currently set to 4 workers × 1 thread, about 2 GB of RAM. The GPU commands use DirectML through `.venv-dml`, a separate venv with `torch-directml` (`pip install torch-directml numpy scipy matplotlib`). Its worker count is set with `DIGITSTUDY_WORKERS`.

```bash
# CPU: B1 equal-steps size sweep + B2 fine-tuning sweep
.venv/Scripts/python -m digitstudy regimes

# GPU, 1 worker (~1.5 GB): finish the validation-only recipe selection, which then writes the headline recipes
DIGITSTUDY_DEVICE=dml DIGITSTUDY_WORKERS=1 .venv-dml/Scripts/python -m digitstudy.domain_gap select

# GPU: final test runs, seeds 0-4 (refuses to start until both headlines exist)
DIGITSTUDY_DEVICE=dml DIGITSTUDY_WORKERS=1 .venv-dml/Scripts/python -m digitstudy.domain_gap final

# reports
.venv/Scripts/python -m digitstudy regimes-report
.venv/Scripts/python -m digitstudy domain-gap-report
.venv/Scripts/python -m digitstudy holm
```

For more speed when RAM is free:
- Regimes: set `"workers": 12` in `configs/regimes.json`. That uses about 5.4 GB and is about 3× faster than 4 workers.
- GPU jobs: use `DIGITSTUDY_WORKERS=3`, about 3.5 GB.

Each domain-gap run records the device that trained it (`"device": "cpu"` or `"dml"`).

After the reports: write the REPORT.md domain-gap section and the restated RQ2 and RQ4 answers, update the README, run `pytest -q` and `python -m digitstudy smoke`, and commit locally. Push only with the user's approval.
