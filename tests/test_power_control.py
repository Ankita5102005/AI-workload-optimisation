import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import actuator.power_control as pc  # noqa: E402


class FakeNVMLError(Exception):
    pass


class FakeNoPermission(FakeNVMLError):
    pass


class FakePynvml:
    """Minimal in-memory stand-in for pynvml, one simulated GPU."""

    def __init__(self, uuid="GPU-fake-0000", limit_w=300.0, default_w=300.0, lo_w=100.0, hi_w=300.0):
        self.uuid, self.limit_w, self.default_w = uuid, limit_w, default_w
        self.lo_w, self.hi_w = lo_w, hi_w
        self.permission_denied = False
        self.NVMLError_NoPermission = FakeNoPermission
        self.NVMLError = FakeNVMLError

    def nvmlInit(self): pass
    def nvmlShutdown(self): pass
    def nvmlDeviceGetHandleByIndex(self, i): return SimpleNamespace(index=i)
    def nvmlDeviceGetUUID(self, h): return self.uuid
    def nvmlDeviceGetPowerManagementLimit(self, h): return self.limit_w * 1000
    def nvmlDeviceGetPowerManagementDefaultLimit(self, h): return self.default_w * 1000
    def nvmlDeviceGetPowerManagementLimitConstraints(self, h): return self.lo_w * 1000, self.hi_w * 1000

    def nvmlDeviceSetPowerManagementLimit(self, h, milliwatts):
        if self.permission_denied:
            raise self.NVMLError_NoPermission("no permission")
        self.limit_w = milliwatts / 1000.0


@pytest.fixture(autouse=True)
def fake_nvml(monkeypatch):
    fake = FakePynvml()
    monkeypatch.setattr(pc, "pynvml", fake)
    return fake


def test_read_limit_and_full_state(fake_nvml):
    assert pc.read_limit_w(0) == 300.0
    st = pc.read_full_state(0)
    assert st.gpu_uuid == "GPU-fake-0000" and st.min_limit_w == 100.0 and st.max_limit_w == 300.0


def test_set_limit_rejects_out_of_range(fake_nvml):
    with pytest.raises(pc.PowerControlError):
        pc.set_limit_w(0, 50.0)   # below min
    with pytest.raises(pc.PowerControlError):
        pc.set_limit_w(0, 400.0)  # above max


def test_set_limit_refuses_wrong_uuid(fake_nvml):
    with pytest.raises(pc.PowerControlError, match="Refusing"):
        pc.set_limit_w(0, 250.0, expect_uuid="GPU-different")
    assert fake_nvml.limit_w == 300.0  # untouched


def test_set_limit_surfaces_permission_error(fake_nvml):
    fake_nvml.permission_denied = True
    with pytest.raises(pc.PowerControlError, match="root"):
        pc.set_limit_w(0, 250.0)


def test_session_restores_on_normal_exit(fake_nvml, tmp_path):
    with pc.PowerLimitSession(0, expect_uuid="GPU-fake-0000", state_dir=tmp_path) as s:
        s.set_w(250.0)
        assert fake_nvml.limit_w == 250.0
        assert (tmp_path / "GPU-fake-0000.json").exists()
    assert fake_nvml.limit_w == 300.0                      # restored
    assert not (tmp_path / "GPU-fake-0000.json").exists()  # state file cleaned up


def test_session_restores_even_on_exception(fake_nvml, tmp_path):
    with pytest.raises(ValueError):
        with pc.PowerLimitSession(0, state_dir=tmp_path) as s:
            s.set_w(200.0)
            raise ValueError("boom")
    assert fake_nvml.limit_w == 300.0
    assert not list(tmp_path.iterdir())


def test_session_refuses_wrong_gpu_before_touching_anything(fake_nvml, tmp_path):
    with pytest.raises(pc.PowerControlError, match="Refusing to start"):
        with pc.PowerLimitSession(0, expect_uuid="GPU-not-this-one", state_dir=tmp_path):
            pass
    assert fake_nvml.limit_w == 300.0
    assert not list(tmp_path.iterdir())


def test_session_refuses_when_state_file_already_present(fake_nvml, tmp_path):
    (tmp_path / "GPU-fake-0000.json").write_text("{}")
    with pytest.raises(pc.PowerControlError, match="already exists"):
        with pc.PowerLimitSession(0, state_dir=tmp_path):
            pass
    assert fake_nvml.limit_w == 300.0  # never touched


def test_session_restores_on_sigterm(fake_nvml, tmp_path):
    import os
    import signal
    with pc.PowerLimitSession(0, state_dir=tmp_path) as s:
        s.set_w(225.0)
        assert fake_nvml.limit_w == 225.0
        with pytest.raises(SystemExit):
            os.kill(os.getpid(), signal.SIGTERM)
    # the SIGTERM handler restored before raising SystemExit, and __exit__ (which
    # also runs, since SystemExit propagates through the `with`) is a harmless no-op
    assert fake_nvml.limit_w == 300.0
