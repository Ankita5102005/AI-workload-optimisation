"""Stage 5: consolidate Stage 3/4 V100 logs into one dataset.

Stage 2 (Colab T4) is deliberately EXCLUDED: different GPU, different power
range (70W cap vs the V100's 100-300W), so mixing it in would teach the model
a GPU-dependent relationship as if it were universal.

Stage 3 (V100, 300W only) and Stage 4 (V100, 100-300W, which also covers 300W)
overlap at 300W -- that overlap is kept as EXTRA repeats at that one power
value (more repeats there is a mild, not a problem), not dropped, since every
row is a real independent measurement.

Output columns, one row per real measurement (not pre-averaged -- the
predictor's own training script decides whether to average or not):
  workload, power_limit_w, batch_size, precision   <- predictor INPUTS
  runtime_s, energy_j                               <- predictor TARGETS
  avg_power_w, throughput_sps, avg_gpu_util, avg_mem_util,
  avg_sm_clock_mhz, avg_mem_clock_mhz               <- kept for analysis/
                                                        Stage 6 characterizer,
                                                        NOT used as predictor
                                                        inputs (unknown ahead
                                                        of running a config)
  source                                             <- which stage/file it came from
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

KEEP_COLUMNS = [
    "workload", "power_limit_w", "batch_size", "precision",
    "runtime_s", "energy_j", "total_samples", "avg_power_w", "throughput_sps",
    "avg_gpu_util", "avg_mem_util", "avg_sm_clock_mhz", "avg_mem_clock_mhz",
]
# total_samples is kept specifically so the predictor can be trained on PER-SAMPLE
# runtime/energy (see train_predictor.py) rather than absolute values. The source
# sweeps all used one fixed total_samples per workload (e.g. 8192 for resnet18), so
# a model trained on absolute runtime_s/energy_j implicitly bakes in that one sample
# count and silently mispredicts for any other window size -- confirmed for real: the
# Stage 6 online controller uses a different total_samples per window than the
# sweeps did, and its predictions were off by almost exactly that ratio.


def _load_one(path: Path, power_col: str = "power_limit_w") -> pd.DataFrame:
    df = pd.read_csv(path)
    if "failed" in df.columns:
        n_failed = int(df["failed"].sum())
        if n_failed:
            print(f"  {path.name}: dropping {n_failed} failed row(s)")
        df = df[~df["failed"]].copy()
    if power_col != "power_limit_w":
        df = df.rename(columns={power_col: "power_limit_w"})
    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing expected columns: {missing}")
    df = df[KEEP_COLUMNS].copy()
    df["source"] = path.stem
    return df


def consolidate(paths: Iterable[Path], out_path: Path) -> pd.DataFrame:
    """paths: Stage 3/4 CSVs (Stage 4's target_power_w already equals power_limit_w,
    confirmed 0 mismatches on the real data -- so no special-casing needed there)."""
    frames = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            print(f"  {p}: not found, skipping")
            continue
        print(f"Loading {p.name} ...")
        frames.append(_load_one(p))

    if not frames:
        raise ValueError("No input files found -- nothing to consolidate.")

    combined = pd.concat(frames, ignore_index=True)
    before = len(combined)
    # Exact duplicate measurements (same config, same source, identical metrics) can
    # happen if a file is accidentally fed in twice; a genuine repeat always differs
    # slightly in runtime/energy, so this only removes true accidental duplicates.
    combined = combined.drop_duplicates()
    if len(combined) != before:
        print(f"  Dropped {before - len(combined)} exact-duplicate row(s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)
    print(f"\nConsolidated {len(combined)} rows from {len(frames)} file(s) -> {out_path}")
    print(combined.groupby(["source", "workload"]).size().rename("rows"))
    return combined


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", default=[
        "data/stage3_v100_baseline.csv", "data/stage4_power_sweep.csv",
    ])
    ap.add_argument("--out", default="data/raw_sweep_results.csv")
    args = ap.parse_args()
    consolidate([Path(p) for p in args.inputs], Path(args.out))
