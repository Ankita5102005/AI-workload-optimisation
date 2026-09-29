#!/usr/bin/env python3
"""Crash recovery: find and fix any GPU left at a non-original power limit.

Run this before starting a new session, and any time PowerLimitSession refuses to
start because a state file already exists. It is safe to run even if nothing is
wrong -- it only acts when a leftover file says a GPU's limit differs from what it
should be.

    python scripts/reconcile_power.py                # check every state file
    python scripts/reconcile_power.py --gpu-index 0   # check/fix just this GPU
    python scripts/reconcile_power.py --dry-run       # report only, never sets anything
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from actuator.power_control import DEFAULT_STATE_DIR, PowerLimitState, read_limit_w, set_limit_w  # noqa: E402


def reconcile_one(state_path: Path, *, dry_run: bool) -> bool:
    """Returns True if this GPU is now (or already was) consistent."""
    try:
        state = PowerLimitState.from_json(state_path.read_text())
    except Exception as e:
        print(f"  {state_path.name}: could not parse ({e}). Leaving it -- inspect by hand.")
        return False

    current = read_limit_w(state.device_index)
    drift = abs(current - state.original_limit_w)
    print(f"GPU {state.device_index} ({state.gpu_uuid}): recorded original {state.original_limit_w:.0f} W, "
         f"currently {current:.0f} W (recorded at index may be stale -- verify with nvidia-smi -L).")

    if drift <= 0.5:
        print("  Already consistent. Removing stale state file.")
        if not dry_run:
            state_path.unlink()
        return True

    print(f"  MISMATCH: {drift:.0f} W off. This GPU was left changed by an unclean session exit.")
    if dry_run:
        print("  --dry-run: not touching it.")
        return False

    set_limit_w(state.device_index, state.original_limit_w, expect_uuid=state.gpu_uuid)
    state_path.unlink()
    print(f"  Restored to {state.original_limit_w:.0f} W and removed the state file.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state-dir", default=str(DEFAULT_STATE_DIR))
    ap.add_argument("--gpu-index", type=int, help="only check the state file for this device index")
    ap.add_argument("--dry-run", action="store_true", help="report only, never call nvidia-smi to change anything")
    args = ap.parse_args()

    state_dir = Path(args.state_dir)
    if not state_dir.exists():
        print(f"{state_dir} does not exist -- nothing to reconcile.")
        return

    files = sorted(state_dir.glob("*.json"))
    if args.gpu_index is not None:
        files = [f for f in files
                if PowerLimitState.from_json(f.read_text()).device_index == args.gpu_index]

    if not files:
        print("No leftover session state files found. Nothing to do.")
        return

    print(f"Found {len(files)} leftover state file(s):")
    ok = all(reconcile_one(f, dry_run=args.dry_run) for f in files)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
