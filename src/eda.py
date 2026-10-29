"""Phase-2 exploratory plots. Run:  python -m src.eda"""
from __future__ import annotations

import argparse
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

from .config import Config, load_config
from .evaluation import plot_pca_projection
from .feature_extraction import load_cached_dataset
from .preprocessing import load_audio_fixed
from .utils import set_global_seed, setup_logging

logger = logging.getLogger(__name__)


def _first_per_genre(data: dict) -> dict[str, int]:
    meta = data["meta"]
    return {g: int(meta.index[meta["genre"] == g][0]) for g in data["class_names"]}


def plot_class_counts(data: dict, cfg: Config) -> None:
    counts = data["meta"]["genre"].value_counts().reindex(data["class_names"])
    fig, ax = plt.subplots(figsize=(7, 4))
    counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_ylabel("Songs")
    ax.set_title("Samples per genre")
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(cfg.plots_dir / "samples_per_genre.png", dpi=150)
    plt.close(fig)


def plot_waveforms_and_spectrograms(data: dict, cfg: Config) -> None:
    first = _first_per_genre(data)
    names = list(first)
    fig_w, axes_w = plt.subplots(len(names), 1, figsize=(10, 2 * len(names)), sharex=True)
    fig_s, axes_s = plt.subplots(len(names), 1, figsize=(10, 2.4 * len(names)), sharex=True)
    axes_w, axes_s = np.atleast_1d(axes_w), np.atleast_1d(axes_s)
    t = np.arange(cfg.n_samples) / cfg.sample_rate
    for ax_w, ax_s, g in zip(axes_w, axes_s, names):
        y = load_audio_fixed(cfg.data_dir / data["meta"].loc[first[g], "path"], cfg)
        ax_w.plot(t, y, linewidth=0.4)
        ax_w.set_ylabel(g)
        S = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)),
                                    ref=np.max)
        img = librosa.display.specshow(S, sr=cfg.sample_rate, hop_length=cfg.hop_length,
                                       x_axis="time", y_axis="log", ax=ax_s)
        ax_s.set_ylabel(g)
    axes_w[0].set_title("Waveform examples (one song per genre)")
    axes_w[-1].set_xlabel("Time (s)")
    axes_s[0].set_title("Log-frequency spectrograms (dB)")
    fig_s.colorbar(img, ax=axes_s, format="%+2.0f dB")
    fig_w.tight_layout()
    fig_w.savefig(cfg.plots_dir / "waveform_examples.png", dpi=130)
    fig_s.savefig(cfg.plots_dir / "spectrogram_examples.png", dpi=130, bbox_inches="tight")
    plt.close(fig_w)
    plt.close(fig_s)


def plot_mfcc_heatmaps(data: dict, cfg: Config) -> None:
    first = _first_per_genre(data)
    fig, axes = plt.subplots(len(first), 1, figsize=(10, 2.2 * len(first)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (g, i) in zip(axes, first.items()):
        img = librosa.display.specshow(data["X_mfcc"][i], sr=cfg.sample_rate, hop_length=cfg.hop_length,
                                       x_axis="time", ax=ax, cmap="viridis")
        ax.set_ylabel(g)
    axes[0].set_title("MFCC heatmaps (rows = coefficients, columns = time frames)")
    fig.colorbar(img, ax=axes)
    fig.savefig(cfg.plots_dir / "mfcc_heatmaps.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Exploratory data analysis plots.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    cfg.ensure_dirs()
    data = load_cached_dataset(cfg)
    plot_class_counts(data, cfg)
    plot_waveforms_and_spectrograms(data, cfg)
    plot_mfcc_heatmaps(data, cfg)
    X, y, tr, te = data["X"], data["y"], data["train_idx"], data["test_idx"]
    mfcc_summary = X[:, : 2 * cfg.n_mfcc]  # MFCC means + stds only
    plot_pca_projection(mfcc_summary[tr], mfcc_summary[te], y[tr], y[te], data["class_names"],
                        cfg.plots_dir / "pca_raw_mfcc_features.png",
                        "PCA of MFCC summary features (fit on train only)")
    logger.info("EDA plots saved to %s", cfg.plots_dir)


if __name__ == "__main__":
    main()
