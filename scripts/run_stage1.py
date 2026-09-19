#!/usr/bin/env python3
"""Run Stage 1 end to end on CPU: simulate, search, log, plot.

    python scripts/run_stage1.py
    python scripts/run_stage1.py --strategy random --seed 7

Outputs (default: results/stage1/):
    stage1_trials.csv        every configuration tried, accepted/rejected, and why
    stage1_summary.json      baseline vs chosen configuration per workload
    <workload>_energy.png    energy-vs-configuration sanity plots
"""

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.search import (  # noqa: E402
    Config, CSVTrialLogger, constrained_search, grid_candidates, random_candidates,
)
from sim.cpu_simulator import CPUWorkloadSimulator, GPUSpec  # noqa: E402
from sim.plot_stage1 import plot_workload  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep_config.yaml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "stage1"))
    ap.add_argument("--strategy", choices=["grid", "random"], help="override config")
    ap.add_argument("--seed", type=int, help="override config")
    ap.add_argument("-v", "--verbose", action="store_true", help="log every trial to the console")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s1 = cfg["stage1"]
    seed = args.seed if args.seed is not None else s1["seed"]
    strategy = args.strategy or s1["strategy"]
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    sim = CPUWorkloadSimulator(
        GPUSpec(**cfg["gpu"]), noise_std=s1["noise_std"], seed=seed, num_samples=s1["num_samples"],
    )
    baseline_cfg = Config(**cfg["baseline"])
    candidates = (
        grid_candidates(cfg["sweep"]) if strategy == "grid"
        else random_candidates(cfg["sweep"], s1["random_budget"], seed)
    )
    print(f"Strategy: {strategy} | {len(candidates)} candidates | limit {s1['max_slowdown']}x baseline runtime")

    summary = {}
    with CSVTrialLogger(out / "stage1_trials.csv") as trial_log:
        for workload in cfg["workloads"]:
            result = constrained_search(
                lambda c, w=workload: sim.run(w, **c.as_dict()),
                candidates,
                baseline_cfg,
                workload=workload,
                max_slowdown=s1["max_slowdown"],
                repeats=s1["repeats"],
                baseline_repeats=s1["baseline_repeats"],
                logger_=trial_log,
            )
            plot_workload(result, out / f"{workload}_energy.png")

            n_rej = sum(1 for t in result.trials if t.status == "rejected")
            summary[workload] = {
                "baseline": asdict(result.baseline),
                "chosen": asdict(result.best),
                "energy_saving_pct": result.energy_saving_pct,
                "runtime_overhead_pct": result.runtime_overhead_pct,
                "n_trials": len(result.trials),
                "n_rejected": n_rej,
            }
            print(
                f"\n{workload}\n"
                f"  baseline : {result.baseline.config.label():<22} {result.baseline.runtime_s:.3f}s  {result.baseline.energy_j:.0f}J\n"
                f"  chosen   : {result.best.config.label():<22} {result.best.runtime_s:.3f}s  {result.best.energy_j:.0f}J\n"
                f"  energy saved {result.energy_saving_pct:.1f}% | runtime {result.runtime_overhead_pct:+.1f}% "
                f"| {n_rej}/{len(result.trials)} configs rejected"
            )

    with open(out / "stage1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote log, summary and plots to {out}")


if __name__ == "__main__":
    main()
