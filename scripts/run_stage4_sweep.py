#!/usr/bin/env python3
"""Stage 4: automated sweep across power limit x batch size x precision on the V100.

Builds on the Stage 3 baseline sweep (batch size x precision only) by adding power
limit as a third axis, now that scripts/stage4_probe.py has manually confirmed the
GPU stays stable across 300-200 W.

Loop order is OUTER->INNER: power limit, then repeat, then (batch size, precision).
This deliberately minimizes the number of power-limit CHANGES (the operation your
plan flags as the risky one) to one per value, not one per run. Each power value
is set exactly once per repeat, inside a PowerLimitSession, which guarantees the
limit is restored when that value's block ends -- on success, on a config failure,
or on a crash. Order of (batch, precision) WITHIN a power value's block is shuffled
each repeat, same as Stage 2/3, to spread any thermal drift across configs.

A failure on one config is logged and skipped (not fatal) so one bad combination
doesn't throw away an entire power value's data. A genuine unrecoverable error
(e.g. a broken CUDA context) will still propagate and trigger PowerLimitSession's
restore-on-exception path before stopping the script.

    python scripts/run_stage4_sweep.py --gpu-index 0 --expect-uuid <UUID> --smoke
    python scripts/run_stage4_sweep.py --gpu-index 0 --expect-uuid <UUID>
"""

import argparse
import csv
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actuator.power_control import PowerControlError, PowerLimitSession  # noqa: E402
from telemetry.nvml_poller import NVMLPoller  # noqa: E402
from workloads.common import ROW_FIELDS, run_config  # noqa: E402

EXTRA_FIELDS = ["timestamp", "gpu_name", "repeat", "target_power_w", "failed", "error"]
FIELDS = EXTRA_FIELDS + [f for f in ROW_FIELDS if f not in ("timestamp", "gpu_name", "repeat")]


def make_workload(name: str, device: str):
    if name == "resnet18":
        from workloads.resnet_infer import ResNet18Workload
        return ResNet18Workload(device)
    if name == "distilbert":
        from workloads.distilbert_infer import DistilBertWorkload
        return DistilBertWorkload(device)
    raise ValueError(f"unknown workload {name!r}")


class CSVLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="")
        self._w = csv.DictWriter(self._fh, fieldnames=FIELDS)
        if new_file:
            self._w.writeheader(); self._fh.flush()

    def log(self, row: dict) -> None:
        self._w.writerow({k: row.get(k, "") for k in FIELDS}); self._fh.flush()

    def close(self): self._fh.close()
    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def run_power_value(session: PowerLimitSession, target_w: float, workloads: dict, configs: list,
                    repeats: int, total_samples: dict, warmup_batches: int, cooldown_s: float,
                    gpu_name: str, seed: int, logger: CSVLogger) -> None:
    session.set_w(target_w)
    achieved = session.current_w()
    time.sleep(1.0)
    print(f"\n=== Power limit {target_w:.0f} W (achieved {achieved:.0f} W) ===")

    rng = random.Random(seed)
    for rep in range(repeats):
        order = list(configs)
        rng.shuffle(order)
        for workload_name, bs, prec in order:
            wl = workloads[workload_name]
            poller = NVMLPoller(device_index=session.device_index, interval_s=0.02)
            row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "gpu_name": gpu_name, "repeat": rep, "target_power_w": target_w, "failed": False, "error": "",
                  "workload": workload_name, "batch_size": bs, "precision": prec}
            try:
                res = run_config(wl, bs, prec, total_samples[workload_name], poller,
                                 warmup_batches=warmup_batches, power_limit_w=achieved)
                row.update(res.as_row())
                print(f"  [{workload_name}] rep{rep} bs={bs:<4} {prec}: {res.runtime_s:6.2f}s "
                     f"{res.avg_power_w:6.1f}W {res.energy_j:8.1f}J")
            except Exception as e:
                row["failed"], row["error"] = True, f"{type(e).__name__}: {e}"
                print(f"  [{workload_name}] rep{rep} bs={bs:<4} {prec}: FAILED -- {row['error']}")
            logger.log(row)
            time.sleep(cooldown_s)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep_config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "stage4_power_sweep.csv"))
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--expect-uuid", required=True)
    ap.add_argument("--workloads", nargs="+")
    ap.add_argument("--power-values", nargs="+", type=float, help="override sweep.power_limit_w")
    ap.add_argument("--batch-sizes", nargs="+", type=int)
    ap.add_argument("--precisions", nargs="+", choices=["fp32", "fp16"])
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--smoke", action="store_true", help="tiny run: 2 power values x 1 config, to verify the pipeline")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s2 = cfg["stage2"]
    workload_names = args.workloads or cfg["workloads"]
    power_values = args.power_values or cfg["sweep"]["power_limit_w"]
    batch_sizes = args.batch_sizes or cfg["sweep"]["batch_size"]
    precisions = args.precisions or cfg["sweep"]["precision"]
    repeats = args.repeats or s2["repeats"]
    totals = {w: s2["total_samples"][w] for w in workload_names}
    cooldown = s2["cooldown_s"]
    if args.smoke:
        power_values, batch_sizes, precisions, repeats = power_values[:2], [32], ["fp32"], 1
        totals = {w: 512 for w in workload_names}
        cooldown = 0.5

    configs = [(w, b, p) for w in workload_names for b in batch_sizes for p in precisions]
    print(f"{len(power_values)} power value(s) x {len(configs)} config(s) x {repeats} repeat(s) "
         f"= {len(power_values) * len(configs) * repeats} total runs")

    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible.")
    device = f"cuda:{args.gpu_index}"
    gpu_name = torch.cuda.get_device_name(args.gpu_index)

    workloads = {}
    for name in workload_names:
        print(f"Loading {name} ...")
        workloads[name] = make_workload(name, device)

    try:
        with PowerLimitSession(args.gpu_index, expect_uuid=args.expect_uuid) as session:
            print(f"Session started on GPU {args.gpu_index}. Will restore to "
                 f"{session.state.original_limit_w:.0f} W when every power value is done.")
            with CSVLogger(args.out) as logger:
                for pw in power_values:
                    run_power_value(session, pw, workloads, configs, repeats, totals,
                                    s2["warmup_batches"], cooldown, gpu_name, s2["seed"], logger)
    except PowerControlError as e:
        sys.exit(f"Refusing to run: {e}")
    finally:
        for wl in workloads.values():
            wl.release()

    print(f"\nDone. Results in {args.out}")


if __name__ == "__main__":
    main()
