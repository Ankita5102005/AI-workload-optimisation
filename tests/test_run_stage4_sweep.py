import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import actuator.power_control as pc  # noqa: E402
import run_stage4_sweep as rs  # noqa: E402
from test_power_control import FakePynvml  # noqa: E402
from test_stage2 import FakeReader, DummyWorkload  # noqa: E402


@pytest.fixture(autouse=True)
def fake_nvml(monkeypatch):
    fake = FakePynvml()
    monkeypatch.setattr(pc, "pynvml", fake)
    monkeypatch.setattr(rs, "NVMLPoller", lambda device_index, interval_s: __import__(
        "telemetry.nvml_poller", fromlist=["NVMLPoller"]).NVMLPoller(reader=FakeReader(), interval_s=interval_s))
    return fake


def test_run_power_value_logs_every_config_and_sets_once(fake_nvml, tmp_path):
    workloads = {"w": DummyWorkload(0.001)}
    configs = [("w", 8, "fp32"), ("w", 8, "fp16")]
    with pc.PowerLimitSession(0, state_dir=tmp_path / "state") as session, \
         rs.CSVLogger(tmp_path / "out.csv") as logger:
        rs.run_power_value(session, 250.0, workloads, configs, repeats=2, total_samples={"w": 64},
                           warmup_batches=1, cooldown_s=0, gpu_name="fake", seed=0, logger=logger)
        assert fake_nvml.limit_w == 250.0  # still set during the block
    assert fake_nvml.limit_w == 300.0      # restored after the session

    import csv
    rows = list(csv.DictReader(open(tmp_path / "out.csv")))
    assert len(rows) == 4  # 2 configs x 2 repeats
    assert all(r["target_power_w"] == "250.0" and r["failed"] == "False" for r in rows)
    assert all(float(r["energy_j"]) > 0 for r in rows)


def test_a_failing_config_is_logged_not_fatal(fake_nvml, tmp_path):
    class Boom(DummyWorkload):
        def prepare(self, precision, batch_size):
            def step(i):
                raise RuntimeError("bad combo")
            return step

    good, bad = DummyWorkload(0.001), Boom()
    good.name, bad.name = "good", "bad"  # distinguish instances in the logged rows
    workloads = {"good": good, "bad": bad}
    configs = [("good", 8, "fp32"), ("bad", 8, "fp32")]
    with pc.PowerLimitSession(0, state_dir=tmp_path / "state") as session, \
         rs.CSVLogger(tmp_path / "out.csv") as logger:
        rs.run_power_value(session, 250.0, workloads, configs, repeats=1, total_samples={"good": 64, "bad": 64},
                           warmup_batches=1, cooldown_s=0, gpu_name="fake", seed=0, logger=logger)
    assert fake_nvml.limit_w == 300.0  # still restored even though one config failed

    import csv
    rows = {r["workload"]: r for r in csv.DictReader(open(tmp_path / "out.csv"))}
    assert rows["good"]["failed"] == "False"
    assert rows["bad"]["failed"] == "True" and "bad combo" in rows["bad"]["error"]
