#!/usr/bin/env python3
"""Stage 2: real PyTorch inference + NVML telemetry, sweeping batch size and precision.

    python scripts/run_stage2.py --smoke        # ~1 minute: check NVML + torch + logging work
    python scripts/run_stage2.py                # full sweep from configs/sweep_config.yaml

Every individual run is appended to a CSV (default data/stage2_gpu_results.csv), so the
file survives a Colab disconnect. The power limit is only READ here, never changed.
"""

import argparse
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from telemetry.nvml_poller import NVMLPoller  # noqa: E402
from workloads.common import CSVResultLogger, run_config  # noqa: E402


def make_workload(name: str, device: str):
    if name == "resnet18":
        from workloads.resnet_infer import ResNet18Workload
        return ResNet18Workload(device)
    if name == "distilbert":
        from workloads.distilbert_infer import DistilBertWorkload
        return DistilBertWorkload(device)
    raise ValueError(f"unknown workload {name!r}")


def measure_idle_power(poller, seconds: float = 3.0, sleep=time.sleep) -> float:
    poller.start()
    sleep(seconds)
    samples = poller.stop()
    return statistics.mean(s.power_w for s in samples)


def run_sweep(workload, configs, poller, logger, *, repeats, total_samples, warmup_batches,
              cooldown_s, power_limit_w, gpu_name, seed=0, burn_in=True, sleep=time.sleep):
    """Run every (batch_size, precision) `repeats` times. Order is reshuffled each repeat so
    slow drift (temperature, clocks) is spread across configs instead of biasing the last ones."""
    rng = random.Random(seed)
    rows = []
    if burn_in:  # unlogged run so the first logged config doesn't start on cold clocks
        bs, prec = configs[0]
        run_config(workload, bs, prec, total_samples, poller,
                   warmup_batches=warmup_batches, power_limit_w=power_limit_w)
    for rep in range(repeats):
        order = list(configs)
        rng.shuffle(order)
        for bs, prec in order:
            res = run_config(workload, bs, prec, total_samples, poller,
                             warmup_batches=warmup_batches, power_limit_w=power_limit_w)
            row = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "gpu_name": gpu_name, "repeat": rep, **res.as_row()}
            rows.append(row)
            logger.log(row)
            print(f"  [{workload.name}] rep {rep} bs={bs:<4} {prec}: {res.runtime_s:7.2f}s  "
                  f"{res.avg_power_w:6.1f}W  {res.energy_j:8.1f}J  {res.throughput_sps:8.1f}/s")
            sleep(cooldown_s)
    return rows


def summarize(rows) -> None:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["workload"], r["batch_size"], r["precision"])].append(r)
    print("\nMedian over repeats (CV = run-to-run variation of runtime; the constraint is only 5%):")
    print(f"{'workload':<11}{'bs':>5} {'prec':<5}{'runtime_s':>10}{'power_W':>9}{'energy_J':>10}"
          f"{'J/1k samples':>14}{'runtime CV%':>13}")
    for (w, bs, prec), g in sorted(groups.items()):
        rt = [x["runtime_s"] for x in g]
        cv = 100 * statistics.pstdev(rt) / statistics.mean(rt) if len(rt) > 1 else math.nan
        e = statistics.median(x["energy_j"] for x in g)
        print(f"{w:<11}{bs:>5} {prec:<5}{statistics.median(rt):>10.2f}"
              f"{statistics.median(x['avg_power_w'] for x in g):>9.1f}{e:>10.1f}"
              f"{1000 * e / g[0]['total_samples']:>14.2f}{cv:>13.1f}")

    devs = [abs(r["energy_j"] - r["energy_counter_j"]) / r["energy_counter_j"]
            for r in rows if r["energy_counter_j"] and not math.isnan(r["energy_counter_j"])]
    if devs:
        print(f"\nIntegrated energy vs NVML hardware counter: mean deviation {100 * statistics.mean(devs):.1f}%, "
              f"max {100 * max(devs):.1f}%" + ("  <-- large; check polling / run length" if max(devs) > 0.10 else ""))
    else:
        print("\nNVML hardware energy counter not available on this GPU (cross-check skipped).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep_config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "stage2_gpu_results.csv"))
    ap.add_argument("--workloads", nargs="+")
    ap.add_argument("--batch-sizes", nargs="+", type=int)
    ap.add_argument("--precisions", nargs="+", choices=["fp32", "fp16"])
    ap.add_argument("--repeats", type=int)
    ap.add_argument("--total-samples", type=int, help="override inferences per run for all workloads")
    ap.add_argument("--smoke", action="store_true", help="tiny run to verify the pipeline")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible. In Colab: Runtime > Change runtime type > select a GPU.")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s2 = cfg["stage2"]
    workloads = args.workloads or cfg["workloads"]
    batch_sizes = args.batch_sizes or cfg["sweep"]["batch_size"]
    precisions = args.precisions or cfg["sweep"]["precision"]
    repeats = args.repeats or s2["repeats"]
    totals = {w: args.total_samples or s2["total_samples"][w] for w in workloads}
    cooldown = s2["cooldown_s"]
    if args.smoke:
        batch_sizes, precisions, repeats, cooldown = [32], ["fp32", "fp16"], 1, 0.5
        totals = {w: 512 for w in workloads}
    configs = [(b, p) for b in batch_sizes for p in precisions]

    poller = NVMLPoller(interval_s=s2["poll_interval_s"])
    reader = poller.reader
    gpu_name, power_limit = reader.gpu_name(), reader.power_limit_w()
    print(f"GPU: {gpu_name} | driver {reader.driver_version()} | power limit {power_limit:.0f} W "
          f"(default {reader.default_power_limit_w():.0f} W) | torch {torch.__version__}")
    idle_w = measure_idle_power(poller)
    print(f"Idle power: {idle_w:.1f} W | {len(configs)} configs x {repeats} repeats per workload")

    out = Path(args.out)
    out.with_suffix(".env.json").write_text(json.dumps({
        "gpu_name": gpu_name, "driver_version": reader.driver_version(),
        "power_limit_w": power_limit, "default_power_limit_w": reader.default_power_limit_w(),
        "idle_power_w": idle_w, "torch": torch.__version__, "cuda": torch.version.cuda,
        "total_samples": totals, "poll_interval_s": s2["poll_interval_s"],
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))

    all_rows = []
    with CSVResultLogger(out) as logger:
        for w in workloads:
            print(f"\nLoading {w} ...")
            wl = make_workload(w, "cuda")
            all_rows += run_sweep(
                wl, configs, poller, logger, repeats=repeats, total_samples=totals[w],
                warmup_batches=s2["warmup_batches"], cooldown_s=cooldown,
                power_limit_w=power_limit, gpu_name=gpu_name, seed=s2["seed"],
            )
            del wl
            torch.cuda.empty_cache()
    poller.close()
    summarize(all_rows)
    print(f"\nWrote {len(all_rows)} runs to {out}")


if __name__ == "__main__":
    main()
