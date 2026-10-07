import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from predictor.model_io import check_meta, load_models, write_meta  # noqa: E402


def test_meta_roundtrip_ok(tmp_path):
    meta = write_meta(tmp_path)
    assert meta["sklearn_version"]
    check_meta(tmp_path)  # same version -> no error


def test_version_mismatch_fails_with_retrain_hint(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"sklearn_version": "0.0.1"}))
    with pytest.raises(RuntimeError, match="Retrain INSIDE the container"):
        check_meta(tmp_path)


def test_missing_models_fail_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="Retrain INSIDE the container"):
        load_models(tmp_path)


def test_corrupt_pickle_gives_clear_error(tmp_path):
    (tmp_path / "runtime_model.joblib").write_bytes(b"not a pickle")
    (tmp_path / "energy_model.joblib").write_bytes(b"not a pickle")
    with pytest.raises(RuntimeError, match="scikit-learn version mismatch"):
        load_models(tmp_path)
