"""Stage 1: synthetic GPU workload simulator (runs on CPU, touches no hardware).

Given a configuration (power limit, batch size, precision) it returns a
plausible runtime, average power draw and energy for a fixed amount of
inference work. The numbers are *not* calibrated to a real V100; they only
need to have realistic trends so that the optimizer logic can be validated:

* Bigger batches raise utilisation (and power draw) with diminishing returns.
* FP16 is faster than FP32 and slightly cheaper in power, so it uses less energy.
* A power limit only bites when it is below what the workload would draw.
  When it does, the GPU clocks down (dynamic power ~ clock^power_exponent), so
  runtime rises. The compute-bound part of the runtime scales with the clock;
  the memory-bound part does not. Raising the limit beyond the workload's
  demand gives no further speedup (diminishing returns).
* Small multiplicative noise makes the output resemble real telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

PRECISIONS = ("fp32", "fp16")


@dataclass(frozen=True)
class GPUSpec:
    tdp_w: float = 250.0
    idle_w: float = 50.0
    min_power_limit_w: float = 100.0
    power_exponent: float = 2.5


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    peak_throughput_sps: float  # samples/s at full utilisation, FP32, uncapped
    batch_half: float           # batch size giving 50% utilisation
    compute_fraction: float     # share of runtime that scales with core clock
    fp16_speedup: float         # throughput multiplier for FP16
    fp16_power_factor: float    # dynamic-power multiplier for FP16


# Rough, plausible profiles. DistilBERT is treated as somewhat more memory-bound
# than ResNet-18, so it tolerates lower power limits better.
PROFILES: Dict[str, WorkloadProfile] = {
    "resnet18": WorkloadProfile("resnet18", 4000.0, 8.0, 0.85, 1.9, 0.90),
    "distilbert": WorkloadProfile("distilbert", 1500.0, 16.0, 0.65, 2.2, 0.92),
}


@dataclass
class SimResult:
    runtime_s: float
    avg_power_w: float
    energy_j: float
    throughput_sps: float


class CPUWorkloadSimulator:
    def __init__(
        self,
        gpu: GPUSpec = GPUSpec(),
        noise_std: float = 0.02,
        seed: Optional[int] = None,
        num_samples: int = 4096,
        profiles: Optional[Dict[str, WorkloadProfile]] = None,
    ):
        self.gpu = gpu
        self.noise_std = noise_std
        self.num_samples = num_samples
        self.profiles = profiles or PROFILES
        self.rng = np.random.default_rng(seed)

    def run(self, workload: str, power_limit_w: float, batch_size: int, precision: str) -> SimResult:
        if workload not in self.profiles:
            raise ValueError(f"unknown workload {workload!r}; known: {sorted(self.profiles)}")
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        g, w = self.gpu, self.profiles[workload]
        # Mirror NVML, which rejects power limits outside the supported range.
        if not (g.min_power_limit_w <= power_limit_w <= g.tdp_w):
            raise ValueError(
                f"power limit {power_limit_w} W outside [{g.min_power_limit_w}, {g.tdp_w}] W"
            )

        fp16 = precision == "fp16"
        util = batch_size / (batch_size + w.batch_half)

        # Uncapped throughput and the time the job would take at full clocks.
        throughput = w.peak_throughput_sps * (w.fp16_speedup if fp16 else 1.0) * util
        t_full = self.num_samples / throughput

        # Power the workload would draw with no cap.
        demand_w = g.idle_w + (g.tdp_w - g.idle_w) * util * (w.fp16_power_factor if fp16 else 1.0)

        if power_limit_w >= demand_w:
            clock_scale = 1.0
        else:
            clock_scale = ((power_limit_w - g.idle_w) / (demand_w - g.idle_w)) ** (1.0 / g.power_exponent)

        runtime = t_full * (w.compute_fraction / clock_scale + (1.0 - w.compute_fraction))
        power = g.idle_w + (demand_w - g.idle_w) * clock_scale ** g.power_exponent

        # Telemetry-like noise (power sensors are less noisy than wall-clock time).
        runtime *= self.rng.lognormal(0.0, self.noise_std)
        power = min(power * self.rng.lognormal(0.0, self.noise_std / 2), power_limit_w)

        return SimResult(
            runtime_s=runtime,
            avg_power_w=power,
            energy_j=power * runtime,
            throughput_sps=self.num_samples / runtime,
        )
