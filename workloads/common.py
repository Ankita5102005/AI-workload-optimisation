"""Shared measurement loop for all workloads (torch-free on purpose, so it is testable).

A workload object must provide:
    name: str
    prepare(precision, batch_size) -> step        # builds model/inputs, returns step(i)
    synchronize()                                 # wait for the GPU to finish queued work
    release()                                     # free per-config resources
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from telemetry.energy import energy_per_inference, summarize_samples


@dataclass
class RunResult:
    workload: str
    batch_size: int
    precision: str
    power_limit_w: float        # limit in force during the run (read-only in Stage 2)
    total_samples: int          # inferences processed in the timed region
    runtime_s: float
    avg_power_w: float
    energy_j: float             # integrated from polled power samples
    energy_counter_j: float     # NVML hardware counter delta (cross-check; NaN if unsupported)
    throughput_sps: float
    energy_per_sample_j: float
    peak_power_w: float
    avg_gpu_util: float
    avg_mem_util: float
    avg_sm_clock_mhz: float
    avg_mem_clock_mhz: float
    n_power_samples: int

    def as_row(self) -> dict:
        return asdict(self)


def run_config(workload, batch_size: int, precision: str, total_samples: int, poller, *,
               warmup_batches: int = 5, power_limit_w: float = math.nan) -> RunResult:
    """Run `total_samples` inferences at one (batch size, precision) and measure them.

    Every timed region is bracketed by a GPU synchronise before the start and end
    timestamps, because CUDA kernel launches are asynchronous.
    """
    if total_samples % batch_size:
        raise ValueError(f"total_samples={total_samples} is not divisible by batch_size={batch_size}; "
                         "every config must process the same amount of work")
    n_batches = total_samples // batch_size

    step = workload.prepare(precision, batch_size)
    started = False
    try:
        for i in range(warmup_batches):      # untimed: cuDNN autotune, allocator, clock ramp-up
            step(i)
        workload.synchronize()

        poller.start()
        started = True
        workload.synchronize()
        t0 = time.perf_counter()
        for i in range(n_batches):
            step(i)
        workload.synchronize()
        t1 = time.perf_counter()
        samples = poller.stop()
        started = False
    finally:
        if started:
            poller.abort()
        workload.release()

    runtime = t1 - t0
    s = summarize_samples(samples, t0, t1)
    return RunResult(
        workload=workload.name, batch_size=batch_size, precision=precision,
        power_limit_w=power_limit_w, total_samples=total_samples,
        runtime_s=runtime, avg_power_w=s["avg_power_w"], energy_j=s["energy_j"],
        energy_counter_j=poller.energy_counter_delta_j,
        throughput_sps=total_samples / runtime,
        energy_per_sample_j=energy_per_inference(s["energy_j"], total_samples),
        peak_power_w=s["peak_power_w"], avg_gpu_util=s["avg_gpu_util"], avg_mem_util=s["avg_mem_util"],
        avg_sm_clock_mhz=s["avg_sm_clock_mhz"], avg_mem_clock_mhz=s["avg_mem_clock_mhz"],
        n_power_samples=s["n_power_samples"],
    )


ROW_FIELDS = ["timestamp", "gpu_name", "repeat"] + [f.name for f in fields(RunResult)]


class CSVResultLogger:
    """Appends one row per run; flushes immediately so a Colab disconnect loses nothing."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._fh = open(self.path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=ROW_FIELDS)
        if new_file:
            self._writer.writeheader()
            self._fh.flush()

    def log(self, row: dict) -> None:
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
