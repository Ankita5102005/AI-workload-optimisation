#!/usr/bin/env python3
"""Stage 3 preflight: confirm GPU visibility and NVML telemetry work here
before running anything, and print enough detail to catch a container problem
early. Run this FIRST, before scripts/run_stage2.py, both outside and (if
applicable) inside your Docker container.

    python scripts/check_nvml.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def section(title: str) -> None:
    print(f"\n--- {title} ---")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-index", type=int, default=0,
                    help="physical GPU index to test telemetry on (see the 'All GPUs' section "
                         "below first, and only pick one with 0%% utilization and no other "
                         "process using it).")
    ap.add_argument("--expect-uuid", help="refuse (exit 1) if the GPU at --gpu-index has a different UUID")
    args = ap.parse_args()

    section("Environment")
    try:
        in_container = "docker" in Path("/proc/1/cgroup").read_text()
    except FileNotFoundError:
        in_container = False
    print(f"Inside a container: {in_container}")

    section("All GPUs on this machine (pick an idle one below)")
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
                          "power.draw,power.limit", "--format=csv"],
            capture_output=True, text=True, timeout=10,
        )
        print(out.stdout.strip() or out.stderr.strip())
        if out.returncode != 0:
            print("nvidia-smi failed. If you are inside a container, it was probably started "
                  "without --gpus all (or --runtime=nvidia). Ask whoever manages the container.")
            sys.exit(1)
        print("\nOnly use a GPU with ~0% utilization AND no process other than Xorg/gnome-shell "
              "in `nvidia-smi`'s process list (run plain `nvidia-smi` to see that list). "
              "If every GPU is busy, STOP and wait or ask before proceeding.")
    except FileNotFoundError:
        print("nvidia-smi not found on PATH. GPU driver/tools are not visible from here.")
        sys.exit(1)

    section(f"NVML via pynvml on GPU {args.gpu_index} (what our telemetry poller actually uses)")
    try:
        from telemetry.nvml_poller import NVMLReader
        r = NVMLReader(device_index=args.gpu_index)
        print(f"GPU {args.gpu_index}: {r.gpu_name()} | UUID {r.uuid()} | driver {r.driver_version()}")
        if args.expect_uuid and r.uuid() != args.expect_uuid:
            print(f"WRONG GPU: expected UUID {args.expect_uuid}. Do not proceed.")
            sys.exit(1)
        print(f"Power limit min/max: {r.power_limit_range_w()[0]:.0f}/{r.power_limit_range_w()[1]:.0f} W")
        print(f"Current power limit: {r.power_limit_w():.0f} W | default: {r.default_power_limit_w():.0f} W")
        power, gpu_u, mem_u, sm_clk, mem_clk = r.read()
        print(f"Live sample -> power {power:.1f} W, gpu_util {gpu_u}%, mem_util {mem_u}%, "
              f"sm_clock {sm_clk} MHz, mem_clock {mem_clk} MHz")
        counter = r.energy_counter_j()
        print(f"Hardware energy counter available: {counter is not None}"
              + ("" if counter is None else f" (currently {counter:.0f} J)"))
        r.close()
    except ImportError:
        print("pynvml not installed. Run: pip install nvidia-ml-py")
        sys.exit(1)
    except Exception as e:
        print(f"NVML read failed: {type(e).__name__}: {e}")
        if in_container:
            print("You are inside a container and NVML failed. Common fixes: rerun the container "
                  "with --gpus all, or ask for the NVIDIA Container Toolkit to be enabled.")
        sys.exit(1)

    section("PyTorch")
    try:
        import torch
        print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"torch sees {n} GPU(s), index 0..{n - 1}")
            if args.gpu_index < n:
                torch_name = torch.cuda.get_device_name(args.gpu_index)
                print(f"torch cuda:{args.gpu_index} = {torch_name!r} (NVML said {r.gpu_name()!r})")
                if torch_name != r.gpu_name():
                    print("MISMATCH: NVML and torch disagree on which GPU this index is. Do not "
                          "proceed — check for a CUDA_VISIBLE_DEVICES environment variable.")
                    sys.exit(1)
            else:
                print(f"--gpu-index {args.gpu_index} is out of range for torch ({n} GPU(s) visible).")
    except ImportError:
        print("torch not installed yet.")

    print(f"\nAll checks passed for GPU {args.gpu_index}. Safe to run:\n"
          f"  python scripts/run_stage2.py --gpu-index {args.gpu_index} --smoke")


if __name__ == "__main__":
    main()
