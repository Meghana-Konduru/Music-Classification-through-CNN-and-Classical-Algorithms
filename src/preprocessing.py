"""Audio loading and feature computation.

This module is the SINGLE source of truth for preprocessing: training, feature
caching and ``predict_song.py`` all call the same functions, so they cannot drift.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np

from .config import Config

logger = logging.getLogger(__name__)


class AudioLoadError(RuntimeError):
    """Raised when an audio file is missing content, unreadable or silent."""


def fix_length(y: np.ndarray, n_samples: int) -> np.ndarray:
    """Center-crop long signals, right zero-pad short ones."""
    n = y.shape[0]
    if n == n_samples:
        return y
    if n > n_samples:
        start = (n - n_samples) // 2
        return y[start:start + n_samples]
    return np.pad(y, (0, n_samples - n))


def load_audio_fixed(path: Path | str, cfg: Config) -> np.ndarray:
    """Load -> mono -> resample -> fixed duration -> peak normalise.

    Returns a float32 array of exactly ``cfg.n_samples`` samples.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    try:
        y, _ = librosa.load(path, sr=cfg.sample_rate, mono=True)
    except Exception as exc:  # decoder errors differ across backends
        raise AudioLoadError(f"Could not decode {path.name}: {exc}") from exc

    if y.size == 0 or not np.all(np.isfinite(y)):
        raise AudioLoadError(f"{path.name}: empty or non-finite audio")
    peak = float(np.max(np.abs(y)))
    if peak < 1e-8:
        raise AudioLoadError(f"{path.name}: silent audio")

    if y.size < cfg.n_samples:
        logger.warning("%s is shorter than %.1fs (%.1fs): zero-padding.",
                       path.name, cfg.duration, y.size / cfg.sample_rate)
    elif y.size > cfg.n_samples:
        logger.info("%s is longer than %.1fs: center-cropping.", path.name, cfg.duration)
    y = fix_length(y, cfg.n_samples)
    return (y / peak).astype(np.float32)


def compute_mfcc(y: np.ndarray, cfg: Config) -> np.ndarray:
    """MFCC matrix with shape (n_mfcc, n_frames): rows = coefficients, cols = time."""
    mfcc = librosa.feature.mfcc(
        y=y, sr=cfg.sample_rate, n_mfcc=cfg.n_mfcc, n_fft=cfg.n_fft, hop_length=cfg.hop_length
    )
    expected = (cfg.n_mfcc, cfg.n_frames)
    if mfcc.shape != expected:
        raise ValueError(f"MFCC shape {mfcc.shape} != expected {expected}")
    return mfcc.astype(np.float32)


def handcrafted_feature_names(cfg: Config) -> list[str]:
    names = [f"mfcc{i}_mean" for i in range(cfg.n_mfcc)]
    names += [f"mfcc{i}_std" for i in range(cfg.n_mfcc)]
    names += [f"chroma{i}_mean" for i in range(12)]
    names += [f"chroma{i}_std" for i in range(12)]
    for base in ("spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zcr", "rms"):
        names += [f"{base}_mean", f"{base}_std"]
    return names


def compute_handcrafted(y: np.ndarray, mfcc: np.ndarray, cfg: Config) -> np.ndarray:
    """Fixed-length summary vector (74-d for n_mfcc=20)."""
    stft = np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length))
    sr, n_fft, hop = cfg.sample_rate, cfg.n_fft, cfg.hop_length
    chroma = librosa.feature.chroma_stft(S=stft ** 2, sr=sr, n_fft=n_fft)
    scalar_feats = [
        librosa.feature.spectral_centroid(S=stft, sr=sr, n_fft=n_fft, hop_length=hop),
        librosa.feature.spectral_bandwidth(S=stft, sr=sr, n_fft=n_fft, hop_length=hop),
        librosa.feature.spectral_rolloff(S=stft, sr=sr, n_fft=n_fft, hop_length=hop),
        librosa.feature.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop),
        librosa.feature.rms(S=stft, frame_length=n_fft),
    ]
    parts = [mfcc.mean(axis=1), mfcc.std(axis=1), chroma.mean(axis=1), chroma.std(axis=1)]
    for f in scalar_feats:
        parts += [f.mean(axis=1), f.std(axis=1)]
    vec = np.concatenate(parts).astype(np.float32)
    if vec.shape[0] != cfg.n_handcrafted or not np.all(np.isfinite(vec)):
        raise ValueError(f"Bad handcrafted vector: shape {vec.shape}, finite={np.all(np.isfinite(vec))}")
    return vec


def extract_song_features(path: Path | str, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(mfcc (n_mfcc, n_frames), handcrafted (n_handcrafted,))`` for one file."""
    y = load_audio_fixed(path, cfg)
    mfcc = compute_mfcc(y, cfg)
    return mfcc, compute_handcrafted(y, mfcc, cfg)


# ---- CNN input preparation ---------------------------------------------------------
def compute_mfcc_norm_stats(mfcc_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-coefficient mean/std over TRAIN songs and time. Input: (N, n_mfcc, T)."""
    if mfcc_train.ndim != 3:
        raise ValueError(f"Expected (N, n_mfcc, T), got {mfcc_train.shape}")
    x = mfcc_train.astype(np.float64)
    mean = x.mean(axis=(0, 2))
    std = np.maximum(x.std(axis=(0, 2)), 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def prepare_cnn_input(mfcc: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalise with train-only stats and transpose to Conv1D layout.

    (n_mfcc, T) or (N, n_mfcc, T)  ->  (1, T, n_mfcc) or (N, T, n_mfcc)
    """
    if mfcc.ndim == 2:
        mfcc = mfcc[None]
    if mfcc.ndim != 3 or mfcc.shape[1] != mean.shape[0]:
        raise ValueError(f"MFCC shape {mfcc.shape} incompatible with {mean.shape[0]} coefficients")
    x = (mfcc - mean[None, :, None]) / std[None, :, None]
    return np.ascontiguousarray(np.transpose(x, (0, 2, 1))).astype(np.float32)
