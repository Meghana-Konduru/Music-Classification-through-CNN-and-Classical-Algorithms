"""Two-layer dilated 1-D CNN over MFCC sequences (inspired by the CS229 reference).

Input layout is (time, channels) = (n_frames, n_mfcc); MFCC coefficients act as channels.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from .config import Config
from .utils import save_json

logger = logging.getLogger(__name__)


def best_model_path(cfg: Config) -> Path:
    return cfg.cnn_dir / "best_cnn.keras"


def build_dilated_cnn(input_shape: Tuple[int, int], n_classes: int, cfg: Config) -> keras.Model:
    """Conv(d=2) -> BN -> ReLU -> Pool -> Drop -> Conv(d=4) -> ... -> GAP -> Dense -> Softmax.

    Layer names used later by the feature extractor: ``layer1_relu``, ``layer2_relu``, ``embedding``.
    """
    c = cfg.cnn
    if len(c["filters"]) != 2 or len(c["dilation_rates"]) != 2:
        raise ValueError("This project implements the 2-layer dilated CNN: "
                         "'filters' and 'dilation_rates' need exactly 2 entries.")
    pool_cls = layers.AveragePooling1D if c.get("pool_type", "avg") == "avg" else layers.MaxPooling1D

    inputs = keras.Input(shape=input_shape, name="mfcc_input")
    x = inputs
    for i, (filters, dilation) in enumerate(zip(c["filters"], c["dilation_rates"]), start=1):
        x = layers.Conv1D(filters, c["kernel_size"], dilation_rate=dilation,
                          padding="same", name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"layer{i}_relu")(x)
        x = pool_cls(c["pool_size"], name=f"pool{i}")(x)
        x = layers.Dropout(c["conv_dropout"], name=f"drop{i}")(x)
    x = layers.GlobalAveragePooling1D(name="gap")(x)
    x = layers.Dense(c["dense_units"], activation="relu", name="embedding")(x)
    x = layers.Dropout(c["dense_dropout"], name="drop_embedding")(x)
    outputs = layers.Dense(n_classes, activation="softmax", name="softmax")(x)

    model = keras.Model(inputs, outputs, name="dilated_cnn")
    model.compile(optimizer=keras.optimizers.Adam(c["learning_rate"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def check_cnn_input(X: np.ndarray, cfg: Config) -> None:
    """Verify tensor layout between MFCC extraction and Conv1D."""
    expected = (cfg.n_frames, cfg.n_mfcc)
    if X.ndim != 3 or X.shape[1:] != expected:
        raise ValueError(f"CNN input shape {X.shape} != (N, {expected[0]}, {expected[1]})")
    if not np.all(np.isfinite(X)):
        raise ValueError("CNN input contains NaN/inf")


def fit_cnn(model: keras.Model, X_tr: np.ndarray, y_tr: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray, cfg: Config) -> dict:
    """Train with EarlyStopping / ReduceLROnPlateau / ModelCheckpoint (validation only)."""
    c = cfg.cnn
    cfg.cnn_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=c["early_stopping_patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=c["reduce_lr_patience"], min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(str(best_model_path(cfg)), monitor="val_loss",
                                        save_best_only=True, verbose=0),
    ]
    hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val), epochs=c["epochs"],
                     batch_size=c["batch_size"], callbacks=callbacks, verbose=2)
    history = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    save_json(history, cfg.cnn_dir / "history.json")
    return history


def load_best_cnn(cfg: Config) -> keras.Model:
    path = best_model_path(cfg)
    if not path.is_file():
        raise FileNotFoundError(f"Trained CNN not found: {path}. Run: python train_cnn.py")
    return keras.models.load_model(path)
