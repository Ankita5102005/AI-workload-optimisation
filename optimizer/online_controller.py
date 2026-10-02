"""Stage 6: online controller.

Runs a workload in fixed control windows. After each window:
  1. Classify the workload (characterizer/classify_workload.py) from this
     window's utilization, and narrow the candidate configs accordingly.
  2. Use the Stage 5 predictor models to evaluate every remaining candidate
     WITHOUT running it, reject any whose PREDICTED runtime exceeds
     max_slowdown x baseline, and pick the lowest-PREDICTED-energy survivor.
  3. Compare this window's ACTUAL runtime to baseline. If it exceeds the
     (larger) safety_threshold, the predictor was wrong enough to matter:
     immediately revert to the known-safe baseline config and log the event,
     overriding whatever step 2 chose for the NEXT window.
  4. Apply the chosen config's power limit for the next window via
     actuator.power_control (the only module allowed to touch it).

The baseline is re-measured at the START of THIS run, not taken from an old
file -- a lesson from Stage 2's Colab run, where GPU state drifted over a
session and comparing against a stale baseline would have been misleading.

PowerLimitSession wraps the entire controller run, so the power limit is
restored on any exit -- success, an exception, or a crash -- exactly as in
Stage 4.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import pandas as pd

from actuator.power_control import PowerLimitSession
from characterizer.classify_workload import Signals, classify, narrow_candidates
from optimizer.search import Config
from telemetry.nvml_poller import NVMLPoller
from workloads.common import run_config

logger = logging.getLogger(__name__)


@dataclass
class WindowResult:
    window: int
    config: Config
    label: str
    actual_runtime_s: float
    actual_energy_j: float
    predicted_runtime_s: Optional[float]
    predicted_energy_j: Optional[float]
    slowdown_vs_baseline: float
    safety_triggered: bool
    reverted_to_baseline: bool


@dataclass
class ControllerResult:
    baseline_config: Config
    baseline_runtime_s: float
    windows: List[WindowResult] = field(default_factory=list)

    @property
    def safety_events(self) -> List[WindowResult]:
        return [w for w in self.windows if w.safety_triggered]


def predict(models: dict, workload: str, configs: List[Config]) -> pd.DataFrame:
    """models: {'runtime': fitted_pipeline, 'energy': fitted_pipeline} from Stage 5.
    Returns a DataFrame, one row per config, with predicted_runtime_s/predicted_energy_j."""
    X = pd.DataFrame([{"workload": workload, "precision": c.precision,
                       "power_limit_w": c.power_limit_w, "batch_size": c.batch_size} for c in configs])
    X["predicted_runtime_s"] = models["runtime"].predict(X[["workload", "precision", "power_limit_w", "batch_size"]])
    X["predicted_energy_j"] = models["energy"].predict(X[["workload", "precision", "power_limit_w", "batch_size"]])
    X["config"] = configs
    return X


def choose_config(models: dict, workload: str, candidates: List[Config],
                  baseline_runtime_s: float, max_slowdown: float) -> tuple[Config, pd.Series]:
    """Picks the lowest-predicted-energy candidate whose PREDICTED runtime is within
    max_slowdown x baseline. Falls back to the first candidate (never crashes) if
    every prediction exceeds the limit -- that candidate will likely be rejected by
    the ACTUAL slowdown check next window and the safety watchdog will catch it if
    it's genuinely bad; logging that fallback clearly is the caller's job."""
    preds = predict(models, workload, candidates)
    valid = preds[preds["predicted_runtime_s"] <= max_slowdown * baseline_runtime_s]
    row = (valid if len(valid) else preds).sort_values("predicted_energy_j").iloc[0]
    return row["config"], row


def run_controller(
    workload_obj, workload_name: str, models: dict, candidates: List[Config], baseline_config: Config,
    *, device_index: int, expect_uuid: str, n_windows: int, window_samples: int,
    max_slowdown: float = 1.05, safety_threshold: float = 1.5, cooldown_s: float = 0.2,
    poller_factory: Callable[[], NVMLPoller] = None, sleep: Callable[[float], None] = time.sleep,
) -> ControllerResult:
    """window_samples: TOTAL samples processed per window, fixed across every config
    (not a batch count) -- this is what keeps window-to-window comparisons fair. A
    config with a bigger batch size does FEWER, LARGER batches to process the same
    window_samples, exactly like Stage 2/3/4's total_samples. It must be evenly
    divisible by every candidate's (and the baseline's) batch_size; run_config raises
    a clear error if not, so pick e.g. 2560 (divisible by 8,16,32,64,128,256)."""
    poller_factory = poller_factory or (lambda: NVMLPoller(device_index=device_index, interval_s=0.02))

    bad = [c for c in candidates + [baseline_config] if window_samples % c.batch_size != 0]
    if bad:
        bad_sizes = sorted({c.batch_size for c in bad})
        raise ValueError(f"window_samples={window_samples} is not evenly divisible by batch size(s) "
                         f"{bad_sizes}. Pick a window_samples that is a multiple of every candidate's "
                         f"batch size (e.g. the LCM of the sweep's batch_size list).")

    with PowerLimitSession(device_index, expect_uuid=expect_uuid) as session:
        session.set_w(baseline_config.power_limit_w)
        base_res = run_config(workload_obj, baseline_config.batch_size, baseline_config.precision,
                              window_samples, poller_factory(), warmup_batches=2,
                              power_limit_w=baseline_config.power_limit_w)
        baseline_runtime = base_res.runtime_s
        logger.info("Baseline measured fresh this run: %.3fs at %s", baseline_runtime, baseline_config)

        result = ControllerResult(baseline_config, baseline_runtime)
        current = baseline_config
        # Prediction for `current`, made at the moment it was CHOSEN (previous window's
        # end) -- None for the baseline, since it's measured directly, not predicted.
        pred_runtime, pred_energy = None, None
        for w in range(n_windows):
            if current.power_limit_w != session.current_w():
                session.set_w(current.power_limit_w)
            res = run_config(workload_obj, current.batch_size, current.precision, window_samples,
                             poller_factory(), warmup_batches=1, power_limit_w=current.power_limit_w)
            slowdown = res.runtime_s / baseline_runtime
            label = classify(Signals(res.avg_gpu_util, res.avg_mem_util))

            safety_triggered = slowdown > safety_threshold
            reverted = False
            if safety_triggered:
                logger.warning("Window %d: slowdown %.2fx exceeds safety threshold %.2fx at %s -- "
                              "reverting to baseline", w, slowdown, safety_threshold, current)
                next_config, next_pred_runtime, next_pred_energy = baseline_config, None, None
                reverted = True
            else:
                narrowed = narrow_candidates(label, candidates)
                next_config, row = choose_config(models, workload_name, narrowed, baseline_runtime, max_slowdown)
                next_pred_runtime, next_pred_energy = row["predicted_runtime_s"], row["predicted_energy_j"]

            result.windows.append(WindowResult(
                window=w, config=current, label=label, actual_runtime_s=res.runtime_s,
                actual_energy_j=res.energy_j, predicted_runtime_s=pred_runtime, predicted_energy_j=pred_energy,
                slowdown_vs_baseline=slowdown, safety_triggered=safety_triggered, reverted_to_baseline=reverted,
            ))
            current, pred_runtime, pred_energy = next_config, next_pred_runtime, next_pred_energy
            sleep(cooldown_s)

        if session.current_w() != baseline_config.power_limit_w:
            session.set_w(baseline_config.power_limit_w)

    return result
