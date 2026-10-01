#!/usr/bin/env python3
"""Stage 4: manual power-limit probe, one value at a time, starting near default.

For each value in configs/sweep_config.yaml's `stage4.test_values_w`:
  1. Set the power limit (via actuator.power_control -- the only module allowed to).
  2. Run ONE small real inference (stage4.stability_check) to confirm the GPU still
     responds normally, and to log its runtime/power/energy at this limit.
  3. Move to the next value.
On every exit path (success, a crashed inference, Ctrl+C, SIGTERM) the power limit
is restored to what it was before this script ran -- PowerLimitSession guarantees
this, and it's been verified for real: a normal exit AND a raised exception both
correctly restored 300W on the real V100 (Sep 29 session).

    python scripts/stage4_probe.py --gpu-index 0 --expect-uuid <UUID> --out data/stage4_probe.csv

Run this the same way you ran Stage 3 -- via docker_stage3.sh, --gpu-index 0 --
so telemetry and the actual inference agree on which GPU is being used. The
container now needs --cap-add SYS_ADMIN for power changes to work (already in
docker_stage3.sh).
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actuator.power_control import PowerControlError, PowerLimitSession  # noqa: E402
from telemetry.nvml_poller import NVMLPoller  # noqa: E402
from workloads.common import run_config  # noqa: E402


def make_workload(name: str, device: str):
    if name == "resnet18":
        from workloads.resnet_infer import ResNet18Workload
        return ResNet18Workload(device)
    if name == "distilbert":
        from workloads.distilbert_infer import DistilBertWorkload
        return DistilBertWorkload(device)
    raise ValueError(f"unknown workload {name!r}")


def probe_one_value(session: PowerLimitSession, target_w: float, wl, check_cfg: dict) -> dict:
    """Set target_w, run one small inference, return a result row. Raises if the
    inference itself fails -- PowerLimitSession still restores on the way out."""
    session.set_w(target_w)
    achieved = session.current_w()
    time.sleep(1.0)  # let the GPU settle before measuring

    poller = NVMLPoller(device_index=session.device_index, interval_s=0.02)
    result = run_config(wl, check_cfg["batch_size"], check_cfg["precision"],
                        check_cfg["total_samples"], poller, warmup_batches=2, power_limit_w=achieved)
    return {
        "target_w": target_w, "achieved_w": achieved, "stable": True,
        "runtime_s": result.runtime_s, "avg_power_w": result.avg_power_w,
        "energy_j": result.energy_j, "throughput_sps": result.throughput_sps,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "sweep_config.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "stage4_probe.csv"))
    ap.add_argument("--gpu-index", type=int, default=0, help="0 when run via docker_stage3.sh")
    ap.add_argument("--expect-uuid", required=True, help="refuse to run if this GPU's UUID doesn't match")
    ap.add_argument("--workload", default="resnet18", choices=["resnet18", "distilbert"])
    ap.add_argument("--values", nargs="+", type=float, help="override stage4.test_values_w")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    s4 = cfg["stage4"]
    values = args.values or s4["test_values_w"]
    check_cfg = s4["stability_check"]

    import torch
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU visible.")
    device = f"cuda:{args.gpu_index}"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists() or out.stat().st_size == 0
    fh = open(out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=["target_w", "achieved_w", "stable", "runtime_s",
                                            "avg_power_w", "energy_j", "throughput_sps"])
    if new_file:
        writer.writeheader()

    print(f"Loading {args.workload} ...")
    wl = make_workload(args.workload, device)

    print(f"Probing {len(values)} power values: {values} W, one at a time.")
    try:
        with PowerLimitSession(args.gpu_index, expect_uuid=args.expect_uuid) as session:
            print(f"Session started. Will restore to {session.state.original_limit_w:.0f} W on exit.")
            for v in values:
                print(f"\n--- Setting {v:.0f} W ---")
                try:
                    row = probe_one_value(session, v, wl, check_cfg)
                    print(f"  achieved {row['achieved_w']:.0f} W | runtime {row['runtime_s']:.3f}s | "
                         f"power {row['avg_power_w']:.1f}W | energy {row['energy_j']:.1f}J | STABLE")
                except Exception as e:
                    row = {"target_w": v, "achieved_w": session.current_w(), "stable": False,
                          "runtime_s": None, "avg_power_w": None, "energy_j": None, "throughput_sps": None}
                    print(f"  FAILED at {v:.0f} W: {type(e).__name__}: {e}")
                    print("  Stopping the probe here. The power limit will still be restored on exit.")
                    writer.writerow(row); fh.flush()
                    break
                writer.writerow(row); fh.flush()
    except PowerControlError as e:
        sys.exit(f"Refusing to run: {e}")
    finally:
        fh.close()
        wl.release()

    print(f"\nWrote results to {out}")


if __name__ == "__main__":
    main()
