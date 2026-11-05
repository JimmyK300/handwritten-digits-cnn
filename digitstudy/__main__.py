"""python -m digitstudy {reproduce,study,report,smoke}"""

from __future__ import annotations

import argparse

from digitstudy.report import build_report
from digitstudy.study import load_study_config, run_study


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="digitstudy", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text, default in [
        ("reproduce", "Run the full study, then build tables and figures.", "configs/study.json"),
        ("study", "Run (or resume) the study only.", "configs/study.json"),
        ("report", "Build tables and figures from existing runs.", "configs/study.json"),
        ("smoke", "Tiny end-to-end run (used by CI).", "configs/smoke.json"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", default=default)
        p.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_study_config(args.config)
    if args.command in ("reproduce", "study", "smoke"):
        run_study(cfg)
    if args.command in ("reproduce", "report", "smoke"):
        build_report(cfg, figures=not args.no_figures)


if __name__ == "__main__":
    main()
