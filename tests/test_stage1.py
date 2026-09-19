import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from optimizer.search import Config, constrained_search, grid_candidates  # noqa: E402
from sim.cpu_simulator import CPUWorkloadSimulator  # noqa: E402

SWEEP = {"power_limit_w": [100, 125, 150, 175, 200, 225, 250], "batch_size": [8, 16, 32, 64, 128, 256],
         "precision": ["fp32", "fp16"]}
BASE = Config(250, 32, "fp32")


def quiet_sim():
    return CPUWorkloadSimulator(noise_std=0.0, seed=0)


@pytest.mark.parametrize("workload", ["resnet18", "distilbert"])
def test_higher_power_limit_reduces_runtime_with_diminishing_returns(workload):
    sim = quiet_sim()
    t = [sim.run(workload, p, 128, "fp32").runtime_s for p in (100, 125, 150, 175, 200, 225, 250)]
    assert all(a >= b - 1e-12 for a, b in zip(t, t[1:]))       # never slower with more power
    assert (t[0] - t[1]) > (t[-2] - t[-1])                     # gains shrink at the top


@pytest.mark.parametrize("workload", ["resnet18", "distilbert"])
def test_fp16_faster_and_more_energy_efficient(workload):
    sim = quiet_sim()
    for b in (8, 32, 128):
        fp32, fp16 = sim.run(workload, 250, b, "fp32"), sim.run(workload, 250, b, "fp16")
        assert fp16.runtime_s < fp32.runtime_s and fp16.energy_j < fp32.energy_j


def test_power_never_exceeds_limit_and_invalid_inputs_rejected():
    sim = CPUWorkloadSimulator(noise_std=0.05, seed=1)
    assert all(sim.run("resnet18", 100, 256, "fp32").avg_power_w <= 100 for _ in range(50))
    with pytest.raises(ValueError):
        sim.run("resnet18", 500, 32, "fp32")
    with pytest.raises(ValueError):
        sim.run("resnet18", 250, 32, "int8")


@pytest.mark.parametrize("workload", ["resnet18", "distilbert"])
def test_search_respects_constraint_and_never_worse_than_baseline(workload):
    sim = CPUWorkloadSimulator(noise_std=0.02, seed=3)
    res = constrained_search(lambda c: sim.run(workload, **c.as_dict()), grid_candidates(SWEEP), BASE,
                             workload=workload, max_slowdown=1.05)
    assert res.best.slowdown <= 1.05
    assert res.best.energy_j <= res.baseline.energy_j
    for t in res.trials:
        if t.status == "rejected":
            assert t.slowdown > 1.05
        if t.status == "accepted":
            assert t.slowdown <= 1.05


def test_search_falls_back_to_baseline_when_nothing_is_valid():
    sim = quiet_sim()
    only_slow = [Config(100, 8, "fp32")]
    res = constrained_search(lambda c: sim.run("resnet18", **c.as_dict()), only_slow, BASE, max_slowdown=1.0)
    assert res.best.config == BASE
