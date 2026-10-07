import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictor.consolidate import KEEP_COLUMNS, consolidate  # noqa: E402
from predictor.train_predictor import (  # noqa: E402
    FEATURES_CAT, FEATURES_NUM, add_per_sample_targets, build_pipeline, evaluate_cv,
)

TOTAL_SAMPLES = 1000  # fixed per workload, like the real Stage 2/3/4 sweeps


def make_stage_csv(path, n_power_vals, failed_row=False):
    rows = []
    for w in ["resnet18", "distilbert"]:
        for p in n_power_vals:
            for bs in (32, 64):
                for prec in ("fp32", "fp16"):
                    base = 10.0 / p * bs
                    rows.append({
                        "workload": w, "power_limit_w": p, "batch_size": bs, "precision": prec,
                        "runtime_s": base, "energy_j": base * p, "total_samples": TOTAL_SAMPLES,
                        "avg_power_w": p * 0.9, "throughput_sps": 100 / base, "avg_gpu_util": 50,
                        "avg_mem_util": 20, "avg_sm_clock_mhz": 1000, "avg_mem_clock_mhz": 800, "failed": False,
                    })
    if failed_row:
        rows.append({**rows[0], "runtime_s": -1, "failed": True})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_consolidate_merges_and_drops_failed_rows(tmp_path):
    p1, p2 = tmp_path / "a.csv", tmp_path / "b.csv"
    make_stage_csv(p1, [300])
    make_stage_csv(p2, [100, 200, 300], failed_row=True)
    out = tmp_path / "out.csv"
    combined = consolidate([p1, p2], out)
    assert list(combined.columns) == KEEP_COLUMNS + ["source"]
    assert (combined["runtime_s"] < 0).sum() == 0          # failed row dropped
    assert out.exists()
    assert set(combined["source"].unique()) == {"a", "b"}


def test_consolidate_raises_on_missing_required_column(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"workload": ["resnet18"], "runtime_s": [1.0]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing expected columns"):
        consolidate([p], tmp_path / "out.csv")


def test_consolidate_skips_missing_file_without_crashing(tmp_path, capsys):
    out = tmp_path / "out.csv"
    p1 = tmp_path / "a.csv"
    make_stage_csv(p1, [300])
    consolidate([p1, tmp_path / "does_not_exist.csv"], out)
    assert "not found, skipping" in capsys.readouterr().out


def test_pipeline_learns_a_known_synthetic_relationship(tmp_path):
    p = tmp_path / "a.csv"
    make_stage_csv(p, [100, 150, 200, 250, 300])
    df = add_per_sample_targets(pd.read_csv(p))
    metrics, pred = evaluate_cv(df, "runtime_per_sample_s", n_splits=5, seed=0)
    # runtime_s = 10/power_w * batch_size is a clean, learnable function -- a
    # grouped-CV R2 this high confirms the pipeline (encoding + CV splitting) works
    # end to end, not just that the model object can be constructed.
    assert metrics["r2"] > 0.9
    assert metrics["n_cv_groups"] == df.drop_duplicates(
        subset=["workload", "power_limit_w", "batch_size", "precision"]).shape[0]


def test_per_sample_targets_are_scale_invariant_to_total_samples(tmp_path):
    """The whole point of the fix: the SAME config measured at two different
    total_samples must produce the SAME per-sample target -- this is what makes
    the trained model usable for a control window whose size differs from the
    sweep's. (Directly tests the bug that broke the real Stage 6 controller run.)"""
    p = tmp_path / "a.csv"
    make_stage_csv(p, [300])
    df = pd.read_csv(p)
    doubled = df.copy()
    doubled["total_samples"] *= 2
    doubled["runtime_s"] *= 2   # same per-sample rate, just more total work
    doubled["energy_j"] *= 2
    a = add_per_sample_targets(df)
    b = add_per_sample_targets(doubled)
    pd.testing.assert_series_equal(a["runtime_per_sample_s"], b["runtime_per_sample_s"])
    pd.testing.assert_series_equal(a["energy_per_sample_j"], b["energy_per_sample_j"])


def test_build_pipeline_predicts_without_error():
    pipe = build_pipeline(seed=0)
    df = pd.DataFrame({
        "workload": ["resnet18", "distilbert", "resnet18"],
        "precision": ["fp32", "fp16", "fp16"],
        "power_limit_w": [300, 150, 100],
        "batch_size": [32, 256, 8],
    })
    y = pd.Series([1.0, 2.0, 3.0])
    pipe.fit(df[FEATURES_CAT + FEATURES_NUM], y)
    preds = pipe.predict(df[FEATURES_CAT + FEATURES_NUM])
    assert len(preds) == 3
