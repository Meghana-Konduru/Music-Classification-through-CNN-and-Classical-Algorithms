"""Inference: identical preprocessing to training, predictions from EVERY trained model.

Data flow for one audio file
    wav -> load_audio_fixed -> compute_mfcc ----> compute_handcrafted ------> classical models
                                   |
                                   +-> prepare_cnn_input -> CNN (softmax) ---> CNN prediction
                                                         -> extractors ------> hybrid models
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .classical_models import MODEL_KEYS, model_label
from .cnn_feature_extractor import build_feature_extractors
from .cnn_model import check_cnn_input, load_best_cnn
from .config import Config
from .preprocessing import compute_handcrafted, compute_mfcc, load_audio_fixed, prepare_cnn_input
from .utils import FEATURE_LABELS, load_joblib, load_json

logger = logging.getLogger(__name__)

# CNN representation shown for the hybrid models in the report ('layer1' | 'layer2' | 'embedding').
HYBRID_DISPLAY_SOURCE = "embedding"


def best_model_file(cfg: Config) -> Path:
    return cfg.classical_dir / "best_model.json"


def _load_pipelines(cfg: Config, feature_key: str) -> dict:
    """Load artifacts/classical_models/<feature_key>__<model_key>.joblib for every model type."""
    found = {}
    for mk in MODEL_KEYS:
        path = cfg.classical_dir / f"{feature_key}__{mk}.joblib"
        if path.is_file():
            found[mk] = load_joblib(path)
        else:
            logger.warning("Model file missing, will be skipped: %s", path.name)
    return found


def load_inference_assets(cfg: Config) -> dict:
    """Load the CNN, feature extractors, train-only MFCC stats and all classical pipelines."""
    info_path = cfg.features_dir / "dataset_info.json"
    stats_path = cfg.features_dir / "mfcc_norm_stats.npz"
    if not info_path.is_file() or not stats_path.is_file():
        raise FileNotFoundError("Feature artifacts missing. Run: python -m src.feature_extraction")
    info = load_json(info_path)
    if info["signature"] != cfg.extraction_signature():
        raise ValueError("config.yaml audio/genre settings differ from those used in training.")
    if not best_model_file(cfg).is_file():
        raise FileNotFoundError("No hybrid model selected. Run: python train_hybrid.py")
    best = load_json(best_model_file(cfg))
    stats = np.load(stats_path)
    cnn = load_best_cnn(cfg)

    feature_keys = {"handcrafted", HYBRID_DISPLAY_SOURCE, best["feature_key"]}
    return {
        "class_names": info["class_names"], "mean": stats["mean"], "std": stats["std"],
        "cnn": cnn, "extractors": build_feature_extractors(cnn), "best": best,
        "pipelines": {fk: _load_pipelines(cfg, fk) for fk in sorted(feature_keys)},
    }


def _predict_with_pipeline(pipe, feats: np.ndarray, names: list[str]) -> tuple[str, float | None]:
    """Genre + confidence from predict_proba (all pipelines here support it)."""
    if hasattr(pipe, "predict_proba"):
        probs = pipe.predict_proba(feats)[0]
        k = int(np.argmax(probs))
        return names[int(pipe.classes_[k])], float(probs[k])
    return names[int(pipe.predict(feats)[0])], None


def predict_file(path: Path | str, assets: dict, cfg: Config) -> dict:
    """Run every model on one file and return a structured report.

    Returns {'classical': [...], 'cnn': {...}, 'hybrid': [...], 'best': {...}}
    where each entry is {'name', 'genre', 'confidence'}.
    """
    names = assets["class_names"]
    y = load_audio_fixed(path, cfg)                                   # same loader as training
    mfcc = compute_mfcc(y, cfg)                                       # (n_mfcc, n_frames)
    x = prepare_cnn_input(mfcc, assets["mean"], assets["std"])        # (1, n_frames, n_mfcc)
    check_cnn_input(x, cfg)

    # One matrix per feature source, each fed to the models trained on that source.
    matrices = {"handcrafted": compute_handcrafted(y, mfcc, cfg)[None, :]}
    for key, extractor in assets["extractors"].items():
        matrices[key] = extractor.predict(x, verbose=0)               # layer1 / layer2 / embedding

    def run(feature_key: str, model_key: str) -> dict | None:
        pipe = assets["pipelines"].get(feature_key, {}).get(model_key)
        if pipe is None:
            return None
        genre, conf = _predict_with_pipeline(pipe, matrices[feature_key], names)
        prefix = "" if feature_key == "handcrafted" else f"{FEATURE_LABELS[feature_key]} + "
        return {"name": prefix + model_label(model_key, cfg), "genre": genre, "confidence": conf}

    cnn_probs = assets["cnn"].predict(x, verbose=0)[0]                # softmax output
    k = int(np.argmax(cnn_probs))
    report = {
        "classical": [r for mk in MODEL_KEYS if (r := run("handcrafted", mk))],
        "cnn": {"name": "Dilated CNN", "genre": names[k], "confidence": float(cnn_probs[k])},
        "hybrid": [r for mk in MODEL_KEYS if (r := run(HYBRID_DISPLAY_SOURCE, mk))],
    }
    best = assets["best"]  # chosen by train-CV F1 in train_hybrid.py (never by test score or confidence)
    report["best"] = run(best["feature_key"], best["model_key"])
    report["best"]["cv_f1"] = best["cv_f1"]
    return report
