"""Shared helpers: seeding, logging, JSON / joblib IO, sanity checks."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)

FEATURE_LABELS = {
    "handcrafted": "Handcrafted features",
    "mfcc_summary": "MFCC summary stats",
    "mfcc_sequence": "MFCC sequence",
    "layer1": "CNN Layer 1",
    "layer2": "CNN Layer 2",
    "embedding": "CNN Embedding",
}


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (stdout, timestamped)."""
    logging.basicConfig(
        level=level, stream=sys.stdout, force=True,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%H:%M:%S",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)


def set_global_seed(seed: int = 42, tensorflow: bool = False) -> None:
    """Seed python, numpy and (optionally) TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if tensorflow:
        import tensorflow as tf
        tf.random.set_seed(seed)


def _json_default(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serialisable: {type(obj)}")


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_joblib(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing model file: {path}")
    return joblib.load(path)


def check_feature_matrix(X: np.ndarray, name: str) -> None:
    """Fail early on bad shapes / NaNs / infs."""
    if X.ndim != 2:
        raise ValueError(f"{name}: expected 2-D matrix, got shape {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name}: contains NaN or inf values")
