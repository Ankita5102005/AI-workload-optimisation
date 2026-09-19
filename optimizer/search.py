"""Constrained configuration search (Stage 1, reused for the full sweeps later).

Objective: minimise energy subject to runtime <= max_slowdown x baseline runtime.

The search is independent of where measurements come from. It only needs an
`evaluate_fn(config)` that returns an object with `.runtime_s`, `.avg_power_w`
and `.energy_j`: the CPU simulator in Stage 1, real GPU runs from Stage 2 on.
"""

from __future__ import annotations

import csv
import itertools
import logging
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    power_limit_w: int
    batch_size: int
    precision: str

    def as_dict(self) -> Dict:
        return {"power_limit_w": self.power_limit_w, "batch_size": self.batch_size, "precision": self.precision}

    def label(self) -> str:
        return f"{self.power_limit_w}W/bs{self.batch_size}/{self.precision}"


@dataclass
class Measurement:
    runtime_s: float
    avg_power_w: float
    energy_j: float


@dataclass
class Trial:
    workload: str
    config: Config
    runtime_s: float
    avg_power_w: float
    energy_j: float
    slowdown: float      # runtime / baseline runtime
    energy_ratio: float  # energy / baseline energy
    status: str          # "baseline" | "accepted" | "rejected"
    reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status != "rejected"


@dataclass
class SearchResult:
    workload: str
    baseline: Trial
    best: Trial
    trials: List[Trial]  # includes the baseline as the first entry
    max_slowdown: float

    @property
    def energy_saving_pct(self) -> float:
        return 100.0 * (1.0 - self.best.energy_ratio)

    @property
    def runtime_overhead_pct(self) -> float:
        return 100.0 * (self.best.slowdown - 1.0)


# --------------------------------------------------------------------------- #
# Candidate generation
# --------------------------------------------------------------------------- #
def grid_candidates(sweep: Dict) -> List[Config]:
    return [
        Config(int(p), int(b), str(prec))
        for p, b, prec in itertools.product(sweep["power_limit_w"], sweep["batch_size"], sweep["precision"])
    ]


def random_candidates(sweep: Dict, budget: int, seed: int = 0) -> List[Config]:
    grid = grid_candidates(sweep)
    return random.Random(seed).sample(grid, min(budget, len(grid)))


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_FIELDS = [
    "workload", "power_limit_w", "batch_size", "precision",
    "runtime_s", "avg_power_w", "energy_j",
    "slowdown_vs_baseline", "energy_ratio_vs_baseline", "status", "reason",
]


class CSVTrialLogger:
    """Writes one row per trial, flushed immediately so a crash keeps partial logs."""

    def __init__(self, path, append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not (append and self.path.exists())
        self._fh = open(self.path, "w" if new_file else "a", newline="")
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow(LOG_FIELDS)
            self._fh.flush()

    def log(self, t: Trial) -> None:
        self._writer.writerow([
            t.workload, t.config.power_limit_w, t.config.batch_size, t.config.precision,
            f"{t.runtime_s:.6f}", f"{t.avg_power_w:.4f}", f"{t.energy_j:.4f}",
            f"{t.slowdown:.5f}", f"{t.energy_ratio:.5f}", t.status, t.reason,
        ])
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def measure(evaluate_fn: Callable[[Config], object], config: Config, repeats: int = 1) -> Measurement:
    """Run a configuration `repeats` times and return the per-metric median."""
    runs = [evaluate_fn(config) for _ in range(max(1, repeats))]
    return Measurement(
        runtime_s=statistics.median(r.runtime_s for r in runs),
        avg_power_w=statistics.median(r.avg_power_w for r in runs),
        energy_j=statistics.median(r.energy_j for r in runs),
    )


def constrained_search(
    evaluate_fn: Callable[[Config], object],
    candidates: Iterable[Config],
    baseline_config: Config,
    *,
    workload: str = "workload",
    max_slowdown: float = 1.05,
    repeats: int = 3,
    baseline_repeats: int = 5,
    logger_: Optional[CSVTrialLogger] = None,
) -> SearchResult:
    """Pick the lowest-energy candidate whose runtime <= max_slowdown x baseline.

    The baseline itself is always a valid fallback, so a result always exists.
    """
    if max_slowdown < 1.0:
        raise ValueError("max_slowdown must be >= 1.0")

    base_m = measure(evaluate_fn, baseline_config, baseline_repeats)
    baseline = Trial(
        workload, baseline_config, base_m.runtime_s, base_m.avg_power_w, base_m.energy_j,
        slowdown=1.0, energy_ratio=1.0, status="baseline", reason="reference configuration",
    )
    trials = [baseline]
    best = baseline
    if logger_:
        logger_.log(baseline)
    logger.info("[%s] baseline %s: %.3fs, %.1fJ", workload, baseline_config.label(), base_m.runtime_s, base_m.energy_j)

    for cfg in candidates:
        if cfg == baseline_config:
            continue  # already measured as the baseline
        m = measure(evaluate_fn, cfg, repeats)
        slowdown = m.runtime_s / base_m.runtime_s
        energy_ratio = m.energy_j / base_m.energy_j

        if slowdown > max_slowdown:
            status = "rejected"
            reason = f"runtime {slowdown:.3f}x baseline exceeds {max_slowdown:.2f}x limit"
        else:
            status = "accepted"
            reason = "within runtime limit"

        trial = Trial(workload, cfg, m.runtime_s, m.avg_power_w, m.energy_j, slowdown, energy_ratio, status, reason)
        if trial.accepted and trial.energy_j < best.energy_j:
            best = trial
            trial.reason += "; new best energy"

        trials.append(trial)
        if logger_:
            logger_.log(trial)
        logger.debug("[%s] %s -> %s (%.3fx runtime, %.3fx energy)", workload, cfg.label(), status, slowdown, energy_ratio)

    logger.info("[%s] chosen %s (%.3fx runtime, %.3fx energy)", workload, best.config.label(), best.slowdown, best.energy_ratio)
    return SearchResult(workload, baseline, best, trials, max_slowdown)
