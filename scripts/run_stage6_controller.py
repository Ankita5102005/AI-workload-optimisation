#!/usr/bin/env python3
"""Stage 6: run the online controller on real hardware, using the Stage 5 models.

    python scripts/run_stage6_controller.py --gpu-index 0 --expect-uuid <UUID> \\
        --workload resnet18 --smoke

    python scripts/run_stage6_controller.py --gpu-index 0 --expect-uuid <UUID> \\
        --workload resnet18 --n-windows 20 --window-batches 50

Run via docker_stage3.sh exactly like Stage 3/4 (needs --cap-add SYS_ADMIN for
power-limit control, already in that script).
"""

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optimizer.online_controller import run_controller  # noqa: E402
from optimizer.search import Config, grid_candidates  # noqa: E402


def make_workload(name: str, device: str):
    if name == "resnet18":
        from workloads.resnet_infer import ResNet18Workload
        return ResNet18Workload(device)
    if name == "distilbert":
        from workloads.distilbert_infer import DistilBertWorkload
        return DistilBertWorkload(device)
    raise ValueError(f"unknown workload {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep_config.yaml"))
    ap.add_argument("--models-dir", default=str(ROOT / "predictor" / "model_artifacts"))
    ap.add_argument("--out", default=str(ROOT / "data" / "stage6_controller_run.csv"))
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--expect-uuid", required=True)
    ap.add_argument("--workload", required=True, choices=["resnet18", "distilbert"])
    ap.add_argument("--n-windows", type=int, default=20)
    ap.add_argument("--window-batches", type=int, default=50)
    ap.add_argument("--max-slowdown", type=float, default=1.05)
    ap.add_argument("--safety-threshold", type=float, default=1.5,
                    help="ASSUMPTION: 1.5x baseline runtime triggers an immediate revert to the "
                         "baseline config. The project plan specifies a threshold 'larger' than "
                         "the normal 1.05x constraint but doesn't fix a number -- override this "
                         "if you want something different.")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 3 windows of 5 batches, to verify the pipeline")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    models = {
        "runtime": joblib.load(Path(args.models_dir) / "runtime_model.joblib"),
        "energy": joblib.load(Path(args.models_dir) / "energy_model.joblib"),
    }

    candidates = grid_candidates(cfg["sweep"])
    baseline = Config(**cfg["baseline"])

    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible.")
    device = f"cuda:{args.gpu_index}"
    print(f"Loading {args.workload} ...")
    wl = make_workload(args.workload, device)

    n_windows = 3 if args.smoke else args.n_windows
    window_batches = 5 if args.smoke else args.window_batches

    print(f"Running controller: {n_windows} windows x {window_batches} batches, "
         f"max_slowdown={args.max_slowdown}x, safety_threshold={args.safety_threshold}x")
    try:
        result = run_controller(
            wl, args.workload, models, candidates, baseline,
            device_index=args.gpu_index, expect_uuid=args.expect_uuid,
            n_windows=n_windows, window_batches=window_batches,
            max_slowdown=args.max_slowdown, safety_threshold=args.safety_threshold,
        )
    finally:
        wl.release()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        fields = ["window", "config", "label", "actual_runtime_s", "actual_energy_j",
                 "predicted_runtime_s", "predicted_energy_j", "slowdown_vs_baseline",
                 "safety_triggered", "reverted_to_baseline"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in result.windows:
            d = asdict(row)
            d["config"] = row.config.label()
            w.writerow(d)

    print(f"\nBaseline: {result.baseline_runtime_s:.3f}s at {result.baseline_config.label()}")
    print(f"{len(result.windows)} windows run, {len(result.safety_events)} safety event(s) triggered")
    for ev in result.safety_events:
        print(f"  window {ev.window}: slowdown {ev.slowdown_vs_baseline:.2f}x at {ev.config.label()}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
