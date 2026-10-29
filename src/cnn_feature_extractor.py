"""Turn the trained (frozen) CNN into fixed-length feature extractors.

Taps:
    layer1     post-ReLU output of conv block 1  -> GlobalAveragePooling -> (N, 32)
    layer2     post-ReLU output of conv block 2  -> GlobalAveragePooling -> (N, 64)
    embedding  Dense layer right before softmax                          -> (N, 64)
"""
from __future__ import annotations

import logging

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from .cnn_model import best_model_path, check_cnn_input, load_best_cnn
from .config import Config
from .preprocessing import prepare_cnn_input

logger = logging.getLogger(__name__)

TAP_LAYERS = {"layer1": "layer1_relu", "layer2": "layer2_relu", "embedding": "embedding"}
FEATURE_KEYS = list(TAP_LAYERS)


def build_feature_extractors(model: keras.Model) -> dict[str, keras.Model]:
    """One sub-model per tap; conv taps get global average pooling to fix their length."""
    extractors = {}
    for key, layer_name in TAP_LAYERS.items():
        out = model.get_layer(layer_name).output
        if key in ("layer1", "layer2"):
            out = layers.GlobalAveragePooling1D(name=f"{key}_gap")(out)
        extractors[key] = keras.Model(model.input, out, name=f"extractor_{key}")
    return extractors


def extract_features(extractors: dict[str, keras.Model], X_cnn: np.ndarray,
                     batch_size: int = 64) -> dict[str, np.ndarray]:
    """Run inference-mode (no dropout, BN frozen) feature extraction."""
    feats = {}
    for key, ex in extractors.items():
        f = ex.predict(X_cnn, batch_size=batch_size, verbose=0)
        if f.ndim != 2 or f.shape[0] != X_cnn.shape[0] or not np.all(np.isfinite(f)):
            raise ValueError(f"Bad {key} features: shape {f.shape}")
        feats[key] = f.astype(np.float32)
    return feats


def _paths(cfg: Config, key: str):
    d = cfg.cnn_features_dir
    return d / f"X_{key}_train.npy", d / f"X_{key}_test.npy"


def features_are_fresh(cfg: Config) -> bool:
    """Cached CNN features exist and are newer than the saved CNN weights."""
    model_path = best_model_path(cfg)
    if not model_path.is_file():
        return False
    files = [p for k in FEATURE_KEYS for p in _paths(cfg, k)]
    if not all(p.is_file() for p in files):
        return False
    return min(p.stat().st_mtime for p in files) >= model_path.stat().st_mtime


def extract_and_save(cfg: Config, data: dict, force: bool = False) -> None:
    """Extract features for all songs once (CNN is frozen) and save train/test arrays."""
    if not force and features_are_fresh(cfg):
        logger.info("CNN features are up to date; skipping extraction.")
        return
    model = load_best_cnn(cfg)
    X_cnn = prepare_cnn_input(data["X_mfcc"], data["norm_mean"], data["norm_std"])
    check_cnn_input(X_cnn, cfg)
    feats = extract_features(build_feature_extractors(model), X_cnn)
    tr, te = data["train_idx"], data["test_idx"]
    cfg.cnn_features_dir.mkdir(parents=True, exist_ok=True)
    for key, f in feats.items():
        p_tr, p_te = _paths(cfg, key)
        np.save(p_tr, f[tr])
        np.save(p_te, f[te])
        logger.info("Saved %s features: train %s, test %s", key, f[tr].shape, f[te].shape)
    np.save(cfg.cnn_features_dir / "y_train.npy", data["y"][tr])
    np.save(cfg.cnn_features_dir / "y_test.npy", data["y"][te])


def load_cnn_features(cfg: Config) -> dict[str, dict[str, np.ndarray]]:
    """Return {key: {'train': array, 'test': array}} from disk."""
    out = {}
    for key in FEATURE_KEYS:
        p_tr, p_te = _paths(cfg, key)
        if not (p_tr.is_file() and p_te.is_file()):
            raise FileNotFoundError(f"Missing CNN features for '{key}'. Run: python train_hybrid.py")
        out[key] = {"train": np.load(p_tr), "test": np.load(p_te)}
    return out
