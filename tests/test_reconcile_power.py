import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import actuator.power_control as pc  # noqa: E402
import reconcile_power  # noqa: E402
from test_power_control import FakePynvml  # noqa: E402


@pytest.fixture(autouse=True)
def fake_nvml(monkeypatch):
    fake = FakePynvml()
    monkeypatch.setattr(pc, "pynvml", fake)
    monkeypatch.setattr(reconcile_power, "read_limit_w", pc.read_limit_w)
    monkeypatch.setattr(reconcile_power, "set_limit_w", pc.set_limit_w)
    return fake


def write_state(tmp_path, **overrides):
    st = pc.PowerLimitState(gpu_uuid="GPU-fake-0000", device_index=0, original_limit_w=300.0,
                            min_limit_w=100.0, max_limit_w=300.0, default_limit_w=300.0, recorded_at=0.0)
    st.__dict__.update(overrides)
    p = tmp_path / f"{st.gpu_uuid}.json"
    p.write_text(st.to_json())
    return p


def test_no_state_dir_is_a_noop(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["reconcile_power.py", "--state-dir", str(tmp_path / "missing")])
    reconcile_power.main()
    assert "does not exist" in capsys.readouterr().out


def test_main_end_to_end_restores_and_exits_zero(tmp_path, fake_nvml, monkeypatch):
    fake_nvml.limit_w = 250.0
    write_state(tmp_path, original_limit_w=300.0)
    monkeypatch.setattr(sys, "argv", ["reconcile_power.py", "--state-dir", str(tmp_path)])
    reconcile_power.main()  # should not raise / sys.exit(1)
    assert fake_nvml.limit_w == 300.0
    assert not list(tmp_path.iterdir())


def test_consistent_state_file_is_removed_no_change(tmp_path, fake_nvml, capsys):
    p = write_state(tmp_path)  # fake GPU is already at 300 (== original), consistent
    assert reconcile_power.reconcile_one(p, dry_run=False) is True
    assert not p.exists()
    assert fake_nvml.limit_w == 300.0


def test_drifted_state_is_restored(tmp_path, fake_nvml):
    fake_nvml.limit_w = 200.0  # simulate an unclean exit that left it changed
    p = write_state(tmp_path, original_limit_w=300.0)
    assert reconcile_power.reconcile_one(p, dry_run=False) is True
    assert fake_nvml.limit_w == 300.0
    assert not p.exists()


def test_dry_run_reports_but_never_changes(tmp_path, fake_nvml):
    fake_nvml.limit_w = 200.0
    p = write_state(tmp_path, original_limit_w=300.0)
    assert reconcile_power.reconcile_one(p, dry_run=True) is False
    assert fake_nvml.limit_w == 200.0  # untouched
    assert p.exists()                  # state file kept so a later real run can fix it


def test_malformed_state_file_is_left_alone(tmp_path, fake_nvml):
    p = tmp_path / "GPU-broken.json"
    p.write_text("not json")
    assert reconcile_power.reconcile_one(p, dry_run=False) is False
    assert p.exists()
