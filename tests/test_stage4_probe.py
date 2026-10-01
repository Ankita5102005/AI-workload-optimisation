import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import actuator.power_control as pc  # noqa: E402
import stage4_probe as sp  # noqa: E402
from test_power_control import FakePynvml  # noqa: E402
from test_stage2 import FakeReader, DummyWorkload  # noqa: E402


@pytest.fixture(autouse=True)
def fake_nvml(monkeypatch):
    fake = FakePynvml()
    monkeypatch.setattr(pc, "pynvml", fake)
    return fake


def test_probe_one_value_sets_measures_and_reports(monkeypatch, fake_nvml, tmp_path):
    monkeypatch.setattr(sp, "NVMLPoller", lambda device_index, interval_s: __import__(
        "telemetry.nvml_poller", fromlist=["NVMLPoller"]).NVMLPoller(reader=FakeReader(), interval_s=interval_s))
    with pc.PowerLimitSession(0, state_dir=tmp_path) as session:
        row = sp.probe_one_value(session, 250.0, DummyWorkload(0.002),
                                 {"batch_size": 8, "precision": "fp32", "total_samples": 80})
    assert row["target_w"] == 250.0 and row["achieved_w"] == 250.0 and row["stable"] is True
    assert row["runtime_s"] > 0 and row["energy_j"] > 0
    assert fake_nvml.limit_w == 300.0  # restored after the session


def test_probe_stops_and_still_restores_on_workload_failure(monkeypatch, fake_nvml, tmp_path):
    monkeypatch.setattr(sp, "NVMLPoller", lambda device_index, interval_s: __import__(
        "telemetry.nvml_poller", fromlist=["NVMLPoller"]).NVMLPoller(reader=FakeReader(), interval_s=interval_s))

    class Boom(DummyWorkload):
        def prepare(self, precision, batch_size):
            def step(i):
                raise RuntimeError("simulated crash mid-inference")
            return step

    with pytest.raises(RuntimeError):
        with pc.PowerLimitSession(0, state_dir=tmp_path) as session:
            sp.probe_one_value(session, 250.0, Boom(), {"batch_size": 8, "precision": "fp32", "total_samples": 80})
    assert fake_nvml.limit_w == 300.0  # STILL restored even though the inference crashed
    assert not list(tmp_path.iterdir())
