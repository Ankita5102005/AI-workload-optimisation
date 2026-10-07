#!/usr/bin/env python3
"""Stage 7: final comparisons and figures for the writeup.

For each workload, compares three configurations:
  default      -- 300W/bs32/fp32 (the reference baseline used throughout)
  best_found   -- the lowest-energy config from the Stage 4 full sweep whose
                  MEASURED runtime satisfies the 1.05x constraint (i.e. the
                  offline, exhaustive-search upper bound)
  proposed     -- the config the Stage 6 online controller actually settled on
                  and ran live, with its measured runtime/energy AND its
                  predicted-vs-actual accuracy

Also produces:
  power vs energy, power vs runtime, energy vs runtime  (from the Stage 4 sweep)
  predicted vs actual runtime & energy                  (from the Stage 6 run)

Needs: data/stage4_power_sweep.csv (required) and data/stage6_controller_run.csv
(optional per workload -- a workload with no controller run yet is reported with
default/best_found only, and a note that the controller comparison is pending).

    python scripts/run_stage7_evaluation.py
    python scripts/run_stage7_evaluation.py --controller-csv data/stage6_controller_run_distilbert.csv
"""

import argparse
import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_CONFIG = {"power_limit_w": 300, "batch_size": 32, "precision": "fp32"}
MAX_SLOWDOWN = 1.05
REPORT_SAMPLES = 1000  # everything is reported as "per REPORT_SAMPLES samples"

# IMPORTANT: the Stage 4 sweep and the Stage 6 controller process DIFFERENT total
# sample counts per run (the sweep's total_samples is fixed per workload in
# configs/sweep_config.yaml, e.g. 8192 for resnet18; the controller's window_samples
# is whatever --window-samples was passed, e.g. 2560). Comparing their ABSOLUTE
# runtime_s/energy_j directly is wrong -- same bug class already found and fixed in
# the predictor. Everything below is normalized to PER-SAMPLE, then scaled to
# REPORT_SAMPLES for a human-readable, apples-to-apples number.


def best_found(sweep: pd.DataFrame, workload: str) -> pd.Series:
    g = sweep[sweep.workload == workload].copy()
    g["runtime_per_sample_s"] = g.runtime_s / g.total_samples
    g["energy_per_sample_j"] = g.energy_j / g.total_samples

    base = g[(g.power_limit_w == DEFAULT_CONFIG["power_limit_w"]) &
            (g.batch_size == DEFAULT_CONFIG["batch_size"]) &
            (g.precision == DEFAULT_CONFIG["precision"])]
    base_med = base[["runtime_per_sample_s", "energy_per_sample_j"]].median()

    med = g.groupby(["power_limit_w", "batch_size", "precision"], as_index=False).agg(
        runtime_per_sample_s=("runtime_per_sample_s", "median"),
        energy_per_sample_j=("energy_per_sample_j", "median"))
    valid = med[med.runtime_per_sample_s <= MAX_SLOWDOWN * base_med.runtime_per_sample_s]
    best = (valid if len(valid) else med).sort_values("energy_per_sample_j").iloc[0]
    return pd.Series({
        "default_runtime_s": base_med.runtime_per_sample_s * REPORT_SAMPLES,
        "default_energy_j": base_med.energy_per_sample_j * REPORT_SAMPLES,
        "best_power_w": best.power_limit_w, "best_batch_size": best.batch_size, "best_precision": best.precision,
        "best_runtime_s": best.runtime_per_sample_s * REPORT_SAMPLES,
        "best_energy_j": best.energy_per_sample_j * REPORT_SAMPLES,
    })


def proposed_summary(controller: pd.DataFrame, window_samples_fallback: int = None) -> dict:
    """window 0 is always the fresh baseline measurement (not the controller's choice);
    every later window used the controller's actual chosen config. window_samples is
    read from the CSV's own column when present; window_samples_fallback is only for
    OLDER controller CSVs saved before that column existed (pass --window-samples)."""
    chosen_rows = controller[controller.window > 0].copy()
    if chosen_rows.empty:
        return None
    if "window_samples" in chosen_rows.columns:
        ws = chosen_rows["window_samples"]
    elif window_samples_fallback is not None:
        ws = window_samples_fallback
    else:
        raise ValueError("This controller CSV has no 'window_samples' column (an older run) -- "
                        "pass --window-samples-fallback matching how it was generated.")
    chosen_rows["runtime_per_sample_s"] = chosen_rows.actual_runtime_s / ws
    chosen_rows["energy_per_sample_j"] = chosen_rows.actual_energy_j / ws
    chosen_rows["pred_runtime_per_sample_s"] = chosen_rows.predicted_runtime_s / ws
    chosen_rows["pred_energy_per_sample_j"] = chosen_rows.predicted_energy_j / ws

    pred_err_runtime = (chosen_rows.pred_runtime_per_sample_s - chosen_rows.runtime_per_sample_s).abs() \
        / chosen_rows.runtime_per_sample_s
    pred_err_energy = (chosen_rows.pred_energy_per_sample_j - chosen_rows.energy_per_sample_j).abs() \
        / chosen_rows.energy_per_sample_j
    return {
        "config": chosen_rows.config.mode().iloc[0],  # the config it converged to / ran most often
        "runtime_s": chosen_rows.runtime_per_sample_s.median() * REPORT_SAMPLES,
        "energy_j": chosen_rows.energy_per_sample_j.median() * REPORT_SAMPLES,
        "n_windows": len(chosen_rows),
        "n_safety_events": int(chosen_rows.safety_triggered.sum()),
        "mean_abs_pred_runtime_error_pct": 100 * pred_err_runtime.mean(),
        "mean_abs_pred_energy_error_pct": 100 * pred_err_energy.mean(),
    }


def build_table(sweep: pd.DataFrame, controller_by_workload: dict, window_samples_fallback: int = None) -> pd.DataFrame:
    rows = []
    for workload in sorted(sweep.workload.unique()):
        bf = best_found(sweep, workload)
        row = {
            "workload": workload,
            "default_runtime_s": bf.default_runtime_s, "default_energy_j": bf.default_energy_j,
            "best_found_config": f"{int(bf.best_power_w)}W/bs{int(bf.best_batch_size)}/{bf.best_precision}",
            "best_found_runtime_s": bf.best_runtime_s, "best_found_energy_j": bf.best_energy_j,
            "best_found_energy_saving_pct": 100 * (1 - bf.best_energy_j / bf.default_energy_j),
            "best_found_runtime_overhead_pct": 100 * (bf.best_runtime_s / bf.default_runtime_s - 1),
        }
        ctrl = controller_by_workload.get(workload)
        if ctrl is not None:
            ps = proposed_summary(ctrl, window_samples_fallback)
            if ps:
                row.update({
                    "proposed_config": ps["config"],
                    "proposed_runtime_s": ps["runtime_s"], "proposed_energy_j": ps["energy_j"],
                    "proposed_energy_saving_pct": 100 * (1 - ps["energy_j"] / bf.default_energy_j),
                    "proposed_runtime_overhead_pct": 100 * (ps["runtime_s"] / bf.default_runtime_s - 1),
                    "proposed_n_windows": ps["n_windows"], "proposed_n_safety_events": ps["n_safety_events"],
                    "proposed_mean_abs_pred_runtime_error_pct": ps["mean_abs_pred_runtime_error_pct"],
                    "proposed_mean_abs_pred_energy_error_pct": ps["mean_abs_pred_energy_error_pct"],
                })
        else:
            row["proposed_config"] = "(no controller run yet for this workload)"
        rows.append(row)
    return pd.DataFrame(rows)


def plot_sweep_relationships(sweep: pd.DataFrame, out_path: Path) -> None:
    med = sweep.groupby(["workload", "precision", "power_limit_w", "batch_size"], as_index=False).agg(
        runtime_s=("runtime_s", "median"), avg_power_w=("avg_power_w", "median"), energy_j=("energy_j", "median"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    panels = [("avg_power_w", "energy_j", "Power vs Energy"),
             ("avg_power_w", "runtime_s", "Power vs Runtime"),
             ("runtime_s", "energy_j", "Runtime vs Energy")]
    markers = {"resnet18": "o", "distilbert": "s"}
    colors = {"fp32": "tab:blue", "fp16": "tab:orange"}
    for ax, (xcol, ycol, title) in zip(axes, panels):
        for (w, p), g in med.groupby(["workload", "precision"]):
            ax.scatter(g[xcol], g[ycol], label=f"{w}/{p}", marker=markers.get(w, "x"),
                      color=colors.get(p, "gray"), alpha=0.7, s=28)
        ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(title); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_predicted_vs_actual(controller_by_workload: dict, out_path: Path) -> None:
    all_ctrl = [df.assign(workload=w) for w, df in controller_by_workload.items() if df is not None]
    if not all_ctrl:
        return
    combined = pd.concat(all_ctrl, ignore_index=True)
    combined = combined[combined.window > 0]  # exclude the baseline-only window 0 (no prediction)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (acol, pcol, label) in zip(axes, [("actual_runtime_s", "predicted_runtime_s", "Runtime (s)"),
                                              ("actual_energy_j", "predicted_energy_j", "Energy (J)")]):
        for w, g in combined.groupby("workload"):
            ax.scatter(g[acol], g[pcol], label=w, alpha=0.6, s=22)
        lims = [combined[[acol, pcol]].min().min(), combined[[acol, pcol]].max().max()]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel(f"Actual {label}"); ax.set_ylabel(f"Predicted {label}"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Stage 6 controller: predicted vs. actual (per control window)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_comparison_bars(table: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = range(len(table))
    width = 0.25
    for ax, metric, ylabel in zip(axes, ["runtime_s", "energy_j"], ["Runtime (s)", "Energy (J)"]):
        default_vals = table[f"default_{metric}"]
        best_vals = table[f"best_found_{metric}"]
        ax.bar([i - width for i in x], default_vals, width, label="default")
        ax.bar(x, best_vals, width, label="best found (sweep)")
        if f"proposed_{metric}" in table.columns:
            proposed_vals = table[f"proposed_{metric}"].fillna(0)
            ax.bar([i + width for i in x], proposed_vals, width, label="proposed (controller)")
        ax.set_xticks(list(x)); ax.set_xticklabels(table.workload); ax.set_ylabel(ylabel); ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-csv", default="data/stage4_power_sweep.csv")
    ap.add_argument("--controller-csv", nargs="+", default=["data/stage6_controller_run.csv"],
                    help="one or more Stage 6 controller run CSVs; workload is read from each file's own rows")
    ap.add_argument("--window-samples-fallback", type=int, default=None,
                    help="ONLY for controller CSVs saved before the 'window_samples' column existed -- "
                         "the --window-samples value that run was actually given")
    ap.add_argument("--out-dir", default="paper")
    args = ap.parse_args()

    sweep = pd.read_csv(args.sweep_csv)
    if "failed" in sweep.columns:
        sweep = sweep[~sweep.failed]

    controller_by_workload = {}
    for path in args.controller_csv:
        p = Path(path)
        if not p.exists():
            print(f"  {p}: not found, skipping (that workload's controller comparison will be omitted)")
            continue
        df = pd.read_csv(p)
        if "workload" in df.columns and df["workload"].notna().any():
            workload = df["workload"].dropna().iloc[0]  # newer runs log this directly -- preferred
        else:
            # Older controller CSVs (saved before the 'workload' column existed): fall back to
            # guessing from the file name, e.g. stage6_controller_run_distilbert.csv.
            workload = p.stem.replace("stage6_controller_run_", "").replace("stage6_controller_run", "")
        controller_by_workload[workload or "unknown"] = df

    # If exactly one controller file was given with no workload suffix, and the sweep
    # has exactly one workload with no controller data yet, attach it unambiguously;
    # otherwise the match is left to the --controller-csv file naming convention above.
    if list(controller_by_workload.keys()) == ["unknown"]:
        missing = [w for w in sweep.workload.unique() if w not in controller_by_workload]
        if len(missing) == 1:
            controller_by_workload[missing[0]] = controller_by_workload.pop("unknown")
        else:
            print(f"  Ambiguous workload for the controller CSV with {len(missing)} candidates: {missing}. "
                 f"Name the file stage6_controller_run_<workload>.csv to disambiguate.")

    table = build_table(sweep, controller_by_workload, args.window_samples_fallback)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "stage7_results_table.csv", index=False)
    print(f"\n=== Results table (all runtime/energy figures are PER {REPORT_SAMPLES} SAMPLES -- "
         f"the sweep and the controller used different total sample counts per run, so this is "
         f"the only fair basis for comparison) ===")
    print(table.to_string(index=False))

    plot_sweep_relationships(sweep, out_dir / "stage7_power_energy_runtime.png")
    plot_predicted_vs_actual(controller_by_workload, out_dir / "stage7_predicted_vs_actual.png")
    plot_comparison_bars(table, out_dir / "stage7_comparison_bars.png")

    missing_ctrl = [w for w in sweep.workload.unique() if w not in controller_by_workload]
    if missing_ctrl:
        print(f"\nNOTE: no controller run found for: {missing_ctrl}. Run scripts/run_stage6_controller.py "
             f"for {missing_ctrl} and save its output as data/stage6_controller_run_<workload>.csv, "
             f"then re-run this script for a complete table.")

    with open(out_dir / "stage7_results_table.json", "w") as f:
        json.dump(table.to_dict(orient="records"), f, indent=2, default=str)
    print(f"\nWrote table and figures to {out_dir}/")


if __name__ == "__main__":
    main()
