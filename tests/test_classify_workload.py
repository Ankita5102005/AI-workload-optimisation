import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from characterizer.classify_workload import (  # noqa: E402
    BALANCED, COMPUTE_BOUND, MEMORY_BOUND, Signals, classify, narrow_candidates,
)
from optimizer.search import Config  # noqa: E402

CANDIDATES = [Config(300, b, p) for b in (8, 32, 64, 128, 256) for p in ("fp32", "fp16")]


def test_classify_compute_bound():
    assert classify(Signals(avg_gpu_util=95, avg_mem_util=20)) == COMPUTE_BOUND


def test_classify_memory_bound():
    assert classify(Signals(avg_gpu_util=90, avg_mem_util=85)) == MEMORY_BOUND


def test_classify_balanced():
    assert classify(Signals(avg_gpu_util=50, avg_mem_util=40)) == BALANCED


def test_narrow_memory_bound_excludes_large_batches():
    narrowed = narrow_candidates(MEMORY_BOUND, CANDIDATES)
    assert all(c.batch_size <= 64 for c in narrowed)
    assert len(narrowed) < len(CANDIDATES)


def test_narrow_compute_bound_and_balanced_keep_everything():
    assert narrow_candidates(COMPUTE_BOUND, CANDIDATES) == CANDIDATES
    assert narrow_candidates(BALANCED, CANDIDATES) == CANDIDATES


def test_narrow_never_returns_empty():
    tiny = [Config(300, 256, "fp32")]  # only a large-batch candidate exists
    assert narrow_candidates(MEMORY_BOUND, tiny) == tiny  # falls back rather than emptying
