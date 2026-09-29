#!/usr/bin/env python3
"""Pick an idle GPU on this machine. Since the assigned GPU may differ every SSH
session, nothing downstream may hardcode an index -- this is the first thing to
run each session.

A GPU counts as idle only if:
  - GPU utilization is 0%
  - No process besides Xorg / gnome-shell (or nothing at all) is using it

    python scripts/pick_idle_gpu.py                 # prints "index uuid" of one idle GPU, or exits 1
    python scripts/pick_idle_gpu.py --all            # list every idle GPU, don't pick
    eval "$(python scripts/pick_idle_gpu.py --export)"   # sets $GPU_INDEX and $UUID in the calling shell
"""

import argparse
import subprocess
import sys

ALLOWED_PROCESS_NAMES = {"Xorg", "gnome-shell"}


def query_gpus() -> list[dict]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,utilization.gpu,memory.used",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    if out.returncode != 0:
        sys.exit(f"nvidia-smi failed: {out.stderr.strip()}")
    gpus = []
    for line in out.stdout.strip().splitlines():
        idx, uuid, util, mem = [p.strip() for p in line.split(",")]
        gpus.append({"index": int(idx), "uuid": uuid, "util": int(util), "mem_mib": int(mem), "processes": []})
    return gpus


def query_processes() -> dict[int, list[str]]:
    out = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,name", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10,
    )
    by_uuid: dict[str, list[str]] = {}
    if out.returncode == 0:
        for line in out.stdout.strip().splitlines():
            if not line.strip():
                continue
            uuid, name = [p.strip() for p in line.split(",", 1)]
            by_uuid.setdefault(uuid, []).append(name)
    # --query-compute-apps only lists *compute* processes, which already excludes
    # Xorg/gnome-shell (those are graphics-only). So any entry here is real usage.
    return by_uuid


def is_idle(gpu: dict, procs_by_uuid: dict) -> bool:
    if gpu["util"] != 0:
        return False
    procs = procs_by_uuid.get(gpu["uuid"], [])
    return len(procs) == 0


def find_idle_gpus() -> list[dict]:
    gpus = query_gpus()
    procs = query_processes()
    return [g for g in gpus if is_idle(g, procs)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="list every idle GPU instead of picking one")
    ap.add_argument("--export", action="store_true", help="print `export GPU_INDEX=.. UUID=..` for `eval`")
    args = ap.parse_args()

    idle = find_idle_gpus()
    if not idle:
        print("No idle GPU found. Every GPU has nonzero utilization or an active process. "
             "Wait, or check `nvidia-smi` yourself before proceeding.", file=sys.stderr)
        sys.exit(1)

    if args.all:
        for g in idle:
            print(f"{g['index']} {g['uuid']}")
        return

    chosen = idle[0]
    if args.export:
        print(f"export GPU_INDEX={chosen['index']} UUID={chosen['uuid']}")
    else:
        print(f"{chosen['index']} {chosen['uuid']}")
    if len(idle) > 1:
        print(f"# note: {len(idle)} GPUs were idle; picked index {chosen['index']}", file=sys.stderr)


if __name__ == "__main__":
    main()
