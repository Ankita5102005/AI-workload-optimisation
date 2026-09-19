"""Energy integration and energy-per-inference calculations."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np


def integrate_energy(times: Sequence[float], powers: Sequence[float],
                     t_start: float = None, t_end: float = None) -> float:
    """Trapezoidal integral of power (W) over time (s) -> energy (J).

    If t_start / t_end are given, the integral is limited to that window, with
    the power at the boundaries linearly interpolated between neighbouring samples.
    `times` must be ascending.
    """
    t = np.asarray(times, dtype=float)
    p = np.asarray(powers, dtype=float)
    if t.ndim != 1 or t.shape != p.shape:
        raise ValueError("times and powers must be 1-D arrays of equal length")
    if len(t) < 2:
        raise ValueError("need at least two power samples to integrate")

    lo = t[0] if t_start is None else max(t_start, t[0])
    hi = t[-1] if t_end is None else min(t_end, t[-1])
    if hi <= lo:
        return 0.0

    inside = (t > lo) & (t < hi)
    tt = np.concatenate(([lo], t[inside], [hi]))
    pp = np.concatenate(([np.interp(lo, t, p)], p[inside], [np.interp(hi, t, p)]))
    return float(np.sum(0.5 * (pp[1:] + pp[:-1]) * np.diff(tt)))


def _mean(values) -> float:
    a = np.asarray(list(values), dtype=float)
    a = a[~np.isnan(a)]
    return float(a.mean()) if len(a) else math.nan


def summarize_samples(samples, t_start: float, t_end: float) -> Dict[str, float]:
    """Energy and averaged telemetry for the timed window [t_start, t_end]."""
    duration = t_end - t_start
    if duration <= 0:
        raise ValueError("empty timing window")
    energy = integrate_energy([s.t for s in samples], [s.power_w for s in samples], t_start, t_end)
    inside = [s for s in samples if t_start <= s.t <= t_end] or list(samples)
    return {
        "energy_j": energy,
        "avg_power_w": energy / duration,
        "peak_power_w": max(s.power_w for s in inside),
        "avg_gpu_util": _mean(s.gpu_util for s in inside),
        "avg_mem_util": _mean(s.mem_util for s in inside),
        "avg_sm_clock_mhz": _mean(s.sm_clock_mhz for s in inside),
        "avg_mem_clock_mhz": _mean(s.mem_clock_mhz for s in inside),
        "n_power_samples": len(inside),
    }


def energy_per_inference(energy_j: float, n_inferences: int) -> float:
    return energy_j / n_inferences
