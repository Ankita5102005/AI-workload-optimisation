import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import pick_idle_gpu as pig  # noqa: E402


def run(stdout, returncode=0):
    return SimpleNamespace(stdout=stdout, stderr="", returncode=returncode)


def test_query_gpus_parses_csv(monkeypatch):
    csv = ("0, GPU-aaa, 0, 4215\n1, GPU-bbb, 55, 9702\n2, GPU-ccc, 0, 6\n3, GPU-ddd, 0, 6\n")
    monkeypatch.setattr(pig.subprocess, "run", lambda *a, **k: run(csv))
    gpus = pig.query_gpus()
    assert len(gpus) == 4
    assert gpus[1] == {"index": 1, "uuid": "GPU-bbb", "util": 55, "mem_mib": 9702, "processes": []}


def test_query_gpus_exits_on_nvidia_smi_failure(monkeypatch):
    monkeypatch.setattr(pig.subprocess, "run", lambda *a, **k: run("", returncode=1))
    with pytest.raises(SystemExit):
        pig.query_gpus()


def test_is_idle_requires_zero_util_and_no_compute_process():
    gpu = {"index": 2, "uuid": "GPU-ccc", "util": 0, "mem_mib": 6}
    assert pig.is_idle(gpu, {}) is True
    assert pig.is_idle(gpu, {"GPU-ccc": ["python"]}) is False
    assert pig.is_idle({**gpu, "util": 5}, {}) is False


def test_find_idle_gpus_matches_the_real_scenario(monkeypatch):
    # mirrors the actual nvidia-smi output seen on the DGX Station: GPU 0/1 busy, 2/3 idle
    csv = ("0, GPU-0, 0, 4215\n1, GPU-1, 0, 9702\n2, GPU-2, 0, 6\n3, GPU-3, 0, 6\n")
    procs = "GPU-0, /opt/conda/bin/python\nGPU-1, /opt/conda/bin/python\n"
    calls = iter([run(csv), run(procs)])
    monkeypatch.setattr(pig.subprocess, "run", lambda *a, **k: next(calls))
    idle = pig.find_idle_gpus()
    assert {g["index"] for g in idle} == {2, 3}


def test_main_picks_first_idle_and_prints_index_uuid(monkeypatch, capsys):
    monkeypatch.setattr(pig, "find_idle_gpus", lambda: [{"index": 2, "uuid": "GPU-2"}, {"index": 3, "uuid": "GPU-3"}])
    monkeypatch.setattr(sys, "argv", ["pick_idle_gpu.py"])
    pig.main()
    out = capsys.readouterr()
    assert out.out.splitlines()[0] == "2 GPU-2"


def test_main_exits_nonzero_when_nothing_idle(monkeypatch):
    monkeypatch.setattr(pig, "find_idle_gpus", lambda: [])
    monkeypatch.setattr(sys, "argv", ["pick_idle_gpu.py"])
    with pytest.raises(SystemExit):
        pig.main()


def test_main_export_format(monkeypatch, capsys):
    monkeypatch.setattr(pig, "find_idle_gpus", lambda: [{"index": 3, "uuid": "GPU-3"}])
    monkeypatch.setattr(sys, "argv", ["pick_idle_gpu.py", "--export"])
    pig.main()
    assert capsys.readouterr().out.strip() == "export GPU_INDEX=3 UUID=GPU-3"
