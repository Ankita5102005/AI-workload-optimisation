"""Save/load helpers that make a scikit-learn version mismatch fail LOUDLY and EARLY.

Why this exists: the .joblib models are pickles of scikit-learn objects. A pickle written
by one scikit-learn version (e.g. on the laptop) cannot be reliably read by another
(e.g. the Docker container), and the failure can surface as a confusing error such as
``ModuleNotFoundError: No module named '_loss'``. train_predictor.py now records the
scikit-learn version next to the models (meta.json); load_models() compares it with the
running version and, on any mismatch or load failure, tells you exactly what to do.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

META_NAME = "meta.json"
RETRAIN_HINT = (
    "Retrain INSIDE the container, then re-run:\n"
    "  scripts/docker_stage3.sh $UUID python predictor/consolidate.py\n"
    "  scripts/docker_stage3.sh $UUID python predictor/train_predictor.py"
)


def _minor(v: str) -> str:
    return ".".join(v.split(".")[:2])


def write_meta(out_dir) -> dict:
    import sklearn
    meta = {"sklearn_version": sklearn.__version__, "python": sys.version.split()[0]}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(out_dir) / META_NAME, "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def check_meta(models_dir) -> None:
    """Raise RuntimeError if the saved models were trained with a different scikit-learn."""
    import sklearn
    p = Path(models_dir) / META_NAME
    if not p.exists():
        return  # older artifacts: can't pre-check; load_models() still catches load failures
    saved = json.loads(p.read_text()).get("sklearn_version", "")
    if saved and _minor(saved) != _minor(sklearn.__version__):
        raise RuntimeError(
            f"Models in {models_dir} were trained with scikit-learn {saved}, but this environment "
            f"has {sklearn.__version__}. {RETRAIN_HINT}")


def load_models(models_dir) -> dict:
    import joblib
    models_dir = Path(models_dir)
    check_meta(models_dir)
    try:
        return {
            "runtime": joblib.load(models_dir / "runtime_model.joblib"),
            "energy": joblib.load(models_dir / "energy_model.joblib"),
        }
    except FileNotFoundError as e:
        raise RuntimeError(f"Model file missing in {models_dir}: {e}. {RETRAIN_HINT}") from e
    except Exception as e:  # ModuleNotFoundError('_loss'), AttributeError, pickle errors, ...
        raise RuntimeError(
            f"Could not load models from {models_dir} ({type(e).__name__}: {e}). This is almost "
            f"always a scikit-learn version mismatch between where the models were trained and "
            f"where they are loaded. {RETRAIN_HINT}") from e
