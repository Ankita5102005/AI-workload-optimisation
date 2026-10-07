"""Stage 5: train gradient-boosted regressors predicting PER-SAMPLE runtime and
energy from (workload, power limit, batch size, precision) -- so Stage 6's
controller can evaluate a candidate config, AT WHATEVER TOTAL SAMPLE COUNT IT
ACTUALLY USES, without running it first.

Targets are runtime_per_sample_s and energy_per_sample_j (runtime_s / total_samples,
energy_j / total_samples), NOT the absolute runtime_s / energy_j. This matters: the
source sweeps all used one fixed total_samples per workload, so a model trained on
absolute values implicitly bakes in that one sample count and silently mispredicts
for any other window size -- this broke the Stage 6 controller for real (predictions
were off by ~3.2x, matching the ratio between the sweep's total_samples and the
controller's window size, almost exactly). Per-sample targets are scale-invariant:
the caller multiplies by however many samples their window actually processes.

Inputs are ONLY what's known before a config is actually run: workload,
power_limit_w, batch_size, precision. GPU utilization/clock columns in the
consolidated data are deliberately excluded as inputs (they're only known
AFTER running a config); they stay in the data for Stage 6's workload
characterizer and for analysis, not for this model.

    python predictor/train_predictor.py --data data/raw_sweep_results.csv

Outputs:
  predictor/model_artifacts/runtime_model.joblib
  predictor/model_artifacts/energy_model.joblib
  predictor/model_artifacts/metrics.json
  predictor/model_artifacts/pred_vs_actual.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import GroupKFold, cross_val_predict  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

FEATURES_CAT = ["workload", "precision"]
FEATURES_NUM = ["power_limit_w", "batch_size"]
TARGETS = ["runtime_per_sample_s", "energy_per_sample_j"]


def add_per_sample_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["runtime_per_sample_s"] = df["runtime_s"] / df["total_samples"]
    df["energy_per_sample_j"] = df["energy_j"] / df["total_samples"]
    return df


def build_pipeline(seed: int = 0) -> Pipeline:
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ], remainder="passthrough")
    model = GradientBoostingRegressor(random_state=seed, n_estimators=200, max_depth=3, learning_rate=0.1)
    return Pipeline([("pre", pre), ("model", model)])


def evaluate_cv(df: pd.DataFrame, target: str, n_splits: int, seed: int) -> dict:
    """GroupKFold by config (workload, power, batch, precision), not plain KFold:
    this dataset has repeats of the *same* config, so a random split would leak
    near-identical rows between train and test and overstate accuracy. Grouping
    ensures every repeat of a config stays entirely in one fold."""
    X = df[FEATURES_CAT + FEATURES_NUM]
    y = df[target]
    groups = df["workload"] + "|" + df["power_limit_w"].astype(str) + "|" \
             + df["batch_size"].astype(str) + "|" + df["precision"]
    n_splits = min(n_splits, groups.nunique())
    gkf = GroupKFold(n_splits=n_splits)
    pred = cross_val_predict(build_pipeline(seed), X, y, cv=gkf, groups=groups)
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "n_cv_groups": int(groups.nunique()),
    }, pred


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/raw_sweep_results.csv")
    ap.add_argument("--out-dir", default="predictor/model_artifacts")
    ap.add_argument("--cv-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    required = FEATURES_CAT + FEATURES_NUM + ["runtime_s", "energy_j", "total_samples"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Input data is missing columns: {missing}. If this is an older "
                         f"consolidated file, re-run predictor/consolidate.py (total_samples "
                         f"is now required -- see that file's changelog note).")
    df = add_per_sample_targets(df)
    print(f"Loaded {len(df)} rows from {args.data}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, target in zip(axes, TARGETS):
        print(f"\n=== {target} ===")
        cv_metrics, cv_pred = evaluate_cv(df, target, args.cv_splits, args.seed)
        print(f"  Cross-validated (held-out configs): MAE={cv_metrics['mae']:.3f}  "
             f"RMSE={cv_metrics['rmse']:.3f}  R2={cv_metrics['r2']:.3f}  "
             f"({cv_metrics['n_cv_groups']} distinct configs, {args.cv_splits}-fold grouped)")
        metrics[target] = cv_metrics

        # Final model trained on ALL data (for actual use by Stage 6); the CV score
        # above, not this model's own training error, is what tells you its real accuracy.
        final_model = build_pipeline(args.seed)
        final_model.fit(df[FEATURES_CAT + FEATURES_NUM], df[target])
        name = "runtime" if target == "runtime_per_sample_s" else "energy"
        joblib.dump(final_model, out_dir / f"{name}_model.joblib")

        y = df[target].values
        ax.scatter(y, cv_pred, s=14, alpha=0.5)
        lims = [min(y.min(), cv_pred.min()), max(y.max(), cv_pred.max())]
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlabel(f"Actual {target}"); ax.set_ylabel(f"Predicted {target} (CV)")
        ax.set_title(f"{target}: R2={cv_metrics['r2']:.3f}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "pred_vs_actual.png", dpi=150)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved models, metrics, and plot to {out_dir}/")


if __name__ == "__main__":
    main()
