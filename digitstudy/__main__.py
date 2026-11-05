"""python -m digitstudy {reproduce,study,report,regimes,regimes-report,smoke}

reproduce        dilation study + the five regimes, then all tables and figures
study / report   dilation study only (configs/study.json)
regimes / regimes-report   regimes R1-R5 only (configs/regimes.json)
smoke            tiny end-to-end run of both (used by CI)
"""

from __future__ import annotations

import argparse

from digitstudy import regime_report, regimes
from digitstudy.report import build_report
from digitstudy.study import load_study_config, run_study


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="digitstudy", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["reproduce", "study", "report", "regimes", "regimes-report", "smoke"]:
        p = sub.add_parser(name)
        p.add_argument("--config", default=None, help="dilation-study config")
        p.add_argument("--regimes-config", default=None)
        p.add_argument("--no-figures", action="store_true")
    a = parser.parse_args(argv)
    smoke = a.command == "smoke"
    study_cfg = a.config or ("configs/smoke.json" if smoke else "configs/study.json")
    reg_cfg = a.regimes_config or ("configs/regimes_smoke.json" if smoke else "configs/regimes.json")
    figs = not a.no_figures

    if a.command in ("reproduce", "smoke", "study", "report"):
        cfg = load_study_config(study_cfg)
        if a.command != "report":
            run_study(cfg)
        if a.command != "study":
            build_report(cfg, figures=figs)
    if a.command in ("reproduce", "smoke", "regimes", "regimes-report"):
        rcfg = regimes.load_config(reg_cfg)
        if a.command != "regimes-report":
            regimes.run_regimes(rcfg)
        if a.command != "regimes":
            regime_report.build(rcfg, figs)


if __name__ == "__main__":
    main()
