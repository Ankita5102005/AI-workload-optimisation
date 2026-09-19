"""Stage 2 tests that need no GPU: fake NVML reader + dummy workload."""

import csv
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from telemetry.energy import integrate_energy  # noqa: E402
from telemetry.nvml_poller import NVMLPoller  # noqa: E402
from workloads.common import ROW_FIELDS, CSVResultLogger, run_config  # noqa: E402
import run_stage2  # noqa: E402


class FakeReader:
    """Constant 100 W GPU with a hardware energy counter that tracks wall time."""
    def __init__(self, power_w=100.0):
        self.power_w, self._t0 = power_w, time.perf_counter()

    def read(self):
        return self.power_w, 50.0, 20.0, 1500.0, 5000.0

    def energy_counter_j(self):
        return self.power_w * (time.perf_counter() - self._t0)

    def close(self):
        pass


class DummyWorkload:
    name = "dummy"

    def __init__(self, sleep_s=0.002):
        self.sleep_s, self.released = sleep_s, 0

    def prepare(self, precision, batch_size):
        return lambda i: time.sleep(self.sleep_s)

    def synchronize(self):
        pass

    def release(self):
        self.released += 1


def test_integrate_energy_constant_ramp_and_window():
    assert integrate_energy([0, 1, 2], [100, 100, 100]) == pytest.approx(200.0)
    assert integrate_energy([0, 2], [0, 100]) == pytest.approx(100.0)                   # ramp: triangle
    assert integrate_energy([0, 1, 2], [100, 100, 100], 0.5, 1.5) == pytest.approx(100.0)  # clipped window
    assert integrate_energy([0, 2], [0, 100], 1.0, 2.0) == pytest.approx(75.0)          # interpolated edge
    with pytest.raises(ValueError):
        integrate_energy([0.0], [10.0])


def test_poller_collects_ordered_samples_and_brackets_the_window():
    p = NVMLPoller(reader=FakeReader(), interval_s=0.01)
    t_before = time.perf_counter()
    p.start()
    time.sleep(0.15)
    samples = p.stop()
    t_after = time.perf_counter()
    assert len(samples) >= 8
    assert all(a.t <= b.t for a, b in zip(samples, samples[1:]))
    assert t_before <= samples[0].t and samples[-1].t <= t_after
    assert p.energy_counter_delta_j == pytest.approx(100.0 * 0.15, rel=0.3)


def test_run_config_measures_runtime_power_and_energy():
    wl = DummyWorkload(0.002)
    r = run_config(wl, batch_size=8, precision="fp32", total_samples=80,
                   poller=NVMLPoller(reader=FakeReader(), interval_s=0.01), power_limit_w=70.0)
    assert r.runtime_s >= 10 * 0.002
    assert r.avg_power_w == pytest.approx(100.0, rel=1e-6)
    assert r.energy_j == pytest.approx(100.0 * r.runtime_s, rel=1e-6)
    assert r.energy_per_sample_j == pytest.approx(r.energy_j / 80)
    assert r.throughput_sps == pytest.approx(80 / r.runtime_s)
    assert r.energy_counter_j == pytest.approx(r.energy_j, rel=0.35)
    assert wl.released == 1


def test_run_config_rejects_uneven_work_and_releases_on_error():
    poller = NVMLPoller(reader=FakeReader(), interval_s=0.01)
    with pytest.raises(ValueError):
        run_config(DummyWorkload(), 48, "fp32", 100, poller)

    class Boom(DummyWorkload):
        def prepare(self, precision, batch_size):
            def step(i):
                if i >= 5:  # warm-up passes, timed region fails
                    raise RuntimeError("boom")
            return step

    wl = Boom()
    with pytest.raises(RuntimeError):
        run_config(wl, 8, "fp32", 80, poller, warmup_batches=5)
    assert wl.released == 1 and poller._thread is None   # poller thread was shut down


def test_sweep_logs_every_run_and_summary_prints(tmp_path, capsys):
    out = tmp_path / "res.csv"
    configs = [(8, "fp32"), (8, "fp16"), (16, "fp32")]
    with CSVResultLogger(out) as logger:
        rows = run_stage2.run_sweep(
            DummyWorkload(0.001), configs, NVMLPoller(reader=FakeReader(), interval_s=0.01), logger,
            repeats=2, total_samples=64, warmup_batches=2, cooldown_s=0, power_limit_w=70.0,
            gpu_name="fake", sleep=lambda s: None)
    assert len(rows) == 6                                   # burn-in run is not logged
    with open(out) as f:
        saved = list(csv.DictReader(f))
    assert len(saved) == 6 and list(saved[0].keys()) == ROW_FIELDS
    assert {(int(r["batch_size"]), r["precision"]) for r in saved} == set(configs)
    assert all(float(r["energy_j"]) > 0 for r in saved)
    run_stage2.summarize(rows)
    assert "Median over repeats" in capsys.readouterr().out

    # a second session appends to the same file without repeating the header
    with CSVResultLogger(out) as logger:
        logger.log(rows[0])
    assert len(list(csv.DictReader(open(out)))) == 7
