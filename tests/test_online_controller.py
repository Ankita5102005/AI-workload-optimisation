import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import actuator.power_control as pc  # noqa: E402
from optimizer.online_controller import choose_config, predict, run_controller  # noqa: E402
from optimizer.search import Config  # noqa: E402
from telemetry.nvml_poller import NVMLPoller  # noqa: E402
from test_power_control import FakePynvml  # noqa: E402
from test_stage2 import FakeReader  # noqa: E402

CANDIDATES = [Config(pw, bs, prec) for pw in (300, 200, 100) for bs in (8, 32, 256) for prec in ("fp32", "fp16")]


class StubModel:
    """Mimics the sklearn pipeline .predict(df) interface with a simple closed-form
    function, so controller logic can be tested without a real trained model."""
    def __init__(self, fn):
        self.fn = fn

    def predict(self, X):
        return X.apply(self.fn, axis=1).values


MODELS = {
    # Lower power -> higher runtime (up to a point) and lower energy; bigger batch -> lower runtime.
    "runtime": StubModel(lambda r: 2.0 * (300 / r.power_limit_w) * (32 / r.batch_size) ** 0.5),
    "energy": StubModel(lambda r: r.power_limit_w * 0.01 * (32 / r.batch_size) ** 0.3),
}


class SlowWorkload:
    """Controllable fake: runtime scales with a per-instance factor, so a test can
    force a window to look artificially slow (to trigger the safety watchdog)."""
    name = "fake"

    def __init__(self, slow_after_window=None, base_step_s=0.001):
        self.slow_after_window = slow_after_window
        self.base_step_s = base_step_s
        self._calls = 0

    def prepare(self, precision, batch_size):
        import time
        def step(i):
            time.sleep(self.base_step_s)
        self._calls += 1
        return step

    def synchronize(self): pass
    def release(self): pass


@pytest.fixture(autouse=True)
def fake_nvml(monkeypatch):
    fake = FakePynvml()
    monkeypatch.setattr(pc, "pynvml", fake)
    return fake


def poller_factory():
    return NVMLPoller(reader=FakeReader(power_w=150.0), interval_s=0.01)


def test_predict_and_choose_config_picks_lowest_energy_within_constraint():
    preds = predict(MODELS, "fake", CANDIDATES)
    assert len(preds) == len(CANDIDATES)
    chosen, row = choose_config(MODELS, "fake", CANDIDATES, baseline_runtime_s=2.0, max_slowdown=1.05)
    assert row["predicted_runtime_s"] <= 1.05 * 2.0 + 1e-9
    # among valid candidates, this should be the lowest predicted energy
    valid = [c for c in CANDIDATES if MODELS["runtime"].fn(
        type("R", (), {"power_limit_w": c.power_limit_w, "batch_size": c.batch_size})()) <= 1.05 * 2.0]
    best_energy = min(MODELS["energy"].fn(type("R", (), {"power_limit_w": c.power_limit_w, "batch_size": c.batch_size})())
                      for c in valid)
    assert abs(MODELS["energy"].fn(type("R", (), {"power_limit_w": chosen.power_limit_w,
                                                   "batch_size": chosen.batch_size})()) - best_energy) < 1e-9


def test_choose_config_falls_back_when_nothing_meets_constraint():
    impossible = [Config(300, 8, "fp32")]  # predicted runtime will exceed any tiny baseline
    chosen, row = choose_config(MODELS, "fake", impossible, baseline_runtime_s=0.0001, max_slowdown=1.05)
    assert chosen == impossible[0]  # falls back rather than raising


def test_run_controller_restores_power_and_produces_one_row_per_window(fake_nvml, tmp_path):
    wl = SlowWorkload()
    baseline = Config(300, 32, "fp32")
    result = run_controller(
        wl, "fake", MODELS, CANDIDATES, baseline, device_index=0, expect_uuid=fake_nvml.uuid,
        n_windows=3, window_batches=2, max_slowdown=1.05, safety_threshold=1.5, cooldown_s=0,
        poller_factory=poller_factory, sleep=lambda s: None,
    )
    assert len(result.windows) == 3
    assert result.baseline_runtime_s > 0
    assert fake_nvml.limit_w == 300.0  # restored to baseline/original after the run


def test_safety_watchdog_reverts_on_a_genuinely_slow_window(fake_nvml, tmp_path):
    class OneSlowWindow(SlowWorkload):
        def prepare(self, precision, batch_size):
            import time
            call_n = self._calls
            self._calls += 1
            def step(i):
                # second call (window index 1) runs much slower -> should trip the watchdog
                time.sleep(0.05 if call_n == 1 else 0.001)
            return step

    wl = OneSlowWindow()
    baseline = Config(300, 32, "fp32")
    result = run_controller(
        wl, "fake", MODELS, CANDIDATES, baseline, device_index=0, expect_uuid=fake_nvml.uuid,
        n_windows=3, window_batches=2, max_slowdown=1.05, safety_threshold=1.5, cooldown_s=0,
        poller_factory=poller_factory, sleep=lambda s: None,
    )
    assert any(w.safety_triggered for w in result.windows)
    triggered = [w for w in result.windows if w.safety_triggered][0]
    assert triggered.reverted_to_baseline is True
    assert fake_nvml.limit_w == 300.0  # still restored at the end regardless
