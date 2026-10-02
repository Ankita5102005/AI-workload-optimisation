"""Stage 6: rule-based compute-bound / memory-bound / balanced classifier.

Uses utilization signals from a just-completed control window to label the
workload, and uses that label to narrow which candidate configs the online
controller considers next -- not to pick a config itself.

ASSUMPTIONS (thresholds are heuristic, not derived from the Stage 5 data --
document this in the writeup; tune against real controller runs if the
narrowing looks wrong in practice):
  - high GPU util + low memory util  -> compute-bound: larger batches keep the
    GPU fed without much extra memory traffic, so the full batch-size range is
    worth considering.
  - high memory util                -> memory-bound: increasing batch size
    mostly adds memory pressure, not useful work, so large batches are
    unlikely to help and are excluded from consideration to save evaluation
    time (they remain physically safe, just not worth trying every window).
  - anything in between             -> balanced: no narrowing; consider every
    candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from optimizer.search import Config

COMPUTE_BOUND = "compute_bound"
MEMORY_BOUND = "memory_bound"
BALANCED = "balanced"

# Thresholds (percent, 0-100) -- see module docstring.
HIGH_UTIL = 80.0
HIGH_MEM_UTIL = 80.0
MEMORY_BOUND_MAX_BATCH = 64  # memory-bound: exclude candidates above this batch size


@dataclass(frozen=True)
class Signals:
    avg_gpu_util: float
    avg_mem_util: float


def classify(signals: Signals) -> str:
    if signals.avg_mem_util >= HIGH_MEM_UTIL:
        return MEMORY_BOUND
    if signals.avg_gpu_util >= HIGH_UTIL and signals.avg_mem_util < HIGH_MEM_UTIL:
        return COMPUTE_BOUND
    return BALANCED


def narrow_candidates(label: str, candidates: List[Config]) -> List[Config]:
    """Returns a (possibly) smaller list of configs worth evaluating this window.
    Never returns an empty list if `candidates` is non-empty -- falls back to the
    full list rather than leaving the controller with nothing to choose from."""
    if label == MEMORY_BOUND:
        narrowed = [c for c in candidates if c.batch_size <= MEMORY_BOUND_MAX_BATCH]
        return narrowed or list(candidates)
    return list(candidates)
