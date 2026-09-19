"""Background NVML telemetry poller (Stage 2 onward).

Polls power draw, GPU/memory utilisation and SM/memory clocks at a fixed short
interval on a background thread while a workload runs.

    poller = NVMLPoller(interval_s=0.05)
    poller.start()
    ...run workload...
    samples = poller.stop()

`start()` takes one sample immediately and `stop()` takes one last sample, so the
samples always bracket the timed region and energy integration covers all of it.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import pynvml  # provided by the `nvidia-ml-py` package
except ImportError:  # allows importing this module (and testing) without NVML
    pynvml = None


@dataclass
class Sample:
    t: float               # time.perf_counter() seconds
    power_w: float
    gpu_util: float        # % of time a kernel was running
    mem_util: float        # % of time the memory controller was busy
    sm_clock_mhz: float
    mem_clock_mhz: float


class NVMLReader:
    """Thin wrapper around pynvml for one GPU."""

    def __init__(self, device_index: int = 0):
        if pynvml is None:
            raise RuntimeError("pynvml is not installed. Run: pip install nvidia-ml-py")
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)

    @staticmethod
    def _text(x) -> str:
        return x.decode() if isinstance(x, bytes) else str(x)

    def gpu_name(self) -> str:
        return self._text(pynvml.nvmlDeviceGetName(self.handle))

    def driver_version(self) -> str:
        return self._text(pynvml.nvmlSystemGetDriverVersion())

    def power_limit_w(self) -> float:
        return pynvml.nvmlDeviceGetPowerManagementLimit(self.handle) / 1000.0

    def default_power_limit_w(self) -> float:
        return pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle) / 1000.0

    def read(self) -> Tuple[float, float, float, float, float]:
        power_w = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W (required)
        try:
            u = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            gpu_util, mem_util = float(u.gpu), float(u.memory)
        except pynvml.NVMLError:
            gpu_util = mem_util = math.nan
        try:
            sm = float(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM))
            mem = float(pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM))
        except pynvml.NVMLError:
            sm = mem = math.nan
        return power_w, gpu_util, mem_util, sm, mem

    def energy_counter_j(self) -> Optional[float]:
        """Cumulative hardware energy counter (Volta and newer); None if unsupported."""
        try:
            return pynvml.nvmlDeviceGetTotalEnergyConsumption(self.handle) / 1000.0  # mJ -> J
        except pynvml.NVMLError:
            return None

    def close(self) -> None:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


class NVMLPoller:
    def __init__(self, reader=None, device_index: int = 0, interval_s: float = 0.05):
        self._own_reader = reader is None
        self.reader = reader if reader is not None else NVMLReader(device_index)
        self.interval_s = interval_s
        self.samples: List[Sample] = []
        self.error: Optional[BaseException] = None
        self.energy_counter_start_j: Optional[float] = None
        self.energy_counter_end_j: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- internals ---------------------------------------------------------
    def _sample(self) -> None:
        p, gu, mu, sm, mem = self.reader.read()
        self.samples.append(Sample(time.perf_counter(), p, gu, mu, sm, mem))

    def _counter(self) -> Optional[float]:
        fn = getattr(self.reader, "energy_counter_j", None)
        return fn() if fn else None

    def _loop(self) -> None:
        next_t = time.perf_counter() + self.interval_s
        try:
            while not self._stop.wait(max(0.0, next_t - time.perf_counter())):
                self._sample()
                next_t += self.interval_s
        except BaseException as e:  # keep the error; stop() re-raises it
            self.error = e

    def _halt(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    # -- public API --------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("poller already running")
        self.samples = []
        self.error = None
        self._stop.clear()
        self.energy_counter_start_j = self._counter()
        self._sample()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> List[Sample]:
        self._halt()
        if self.error is not None:
            raise RuntimeError("NVML polling thread failed") from self.error
        self._sample()
        self.energy_counter_end_j = self._counter()
        return self.samples

    def abort(self) -> None:
        """Stop the thread without taking a final sample (used on error paths)."""
        self._halt()

    @property
    def energy_counter_delta_j(self) -> float:
        if self.energy_counter_start_j is None or self.energy_counter_end_j is None:
            return math.nan
        return self.energy_counter_end_j - self.energy_counter_start_j

    def close(self) -> None:
        self.abort()
        if self._own_reader:
            self.reader.close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self.stop()
