"""Scan GTZAN, extract + cache features, and create the ONE shared train/test split.

Run:  python -m src.feature_extraction [--force]

Outputs (artifacts/features/):
    X.npy              (N, 74)            handcrafted summary vectors
    X_mfcc_seq.npy     (N, 20, 1292)      raw MFCC matrices
    y.npy              (N,)               integer labels
    metadata.csv       song_id, path, genre, label, split
    split.json         train/test indices (created once, reused by every script)
    mfcc_norm_stats.npz  per-coefficient mean/std computed on TRAIN songs only
    dataset_info.json  class names, audio settings, feature names, shapes
    dropped_files.csv  files that could not be processed
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import Config, load_config
from .preprocessing import (compute_mfcc_norm_stats, extract_song_features,
                            handcrafted_feature_names)
from .utils import load_json, save_json, set_global_seed, setup_logging

logger = logging.getLogger(__name__)

SETUP_HINT = (
    "Place the GTZAN 'genres_original' folder at {path} so that it contains one sub-folder per "
    "genre ({genres}), each holding .wav files."
)


def scan_dataset(cfg: Config) -> list[tuple[Path, str]]:
    """Return sorted ``(wav_path, genre)`` pairs for the configured genres."""
    if not cfg.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {cfg.data_dir}\n"
                                + SETUP_HINT.format(path=cfg.data_dir, genres=", ".join(cfg.genres)))
    items: list[tuple[Path, str]] = []
    for genre in cfg.genres:
        gdir = cfg.data_dir / genre
        if not gdir.is_dir():
            raise FileNotFoundError(f"Genre folder missing: {gdir}")
        files = sorted(p for p in gdir.iterdir() if p.suffix.lower() == ".wav")
        if not files:
            raise FileNotFoundError(f"No .wav files in {gdir}")
        items.extend((p, genre) for p in files)
    logger.info("Found %d files across %d genres.", len(items), len(cfg.genres))
    return items


def _cache_is_valid(cfg: Config) -> bool:
    fdir = cfg.features_dir
    required = ["X.npy", "X_mfcc_seq.npy", "y.npy", "metadata.csv", "dataset_info.json"]
    if not all((fdir / f).is_file() for f in required):
        return False
    return load_json(fdir / "dataset_info.json").get("signature") == cfg.extraction_signature()


def ensure_split(cfg: Config, force: bool = False) -> dict:
    """Create (or reuse) the stratified song-level split and the train-only MFCC stats."""
    fdir = cfg.features_dir
    y = np.load(fdir / "y.npy")
    split_path = fdir / "split.json"
    if not force and split_path.is_file() and (fdir / "mfcc_norm_stats.npz").is_file():
        s = load_json(split_path)
        if (s.get("test_size") == cfg.test_size and s.get("random_state") == cfg.split_seed
                and s.get("n_samples") == len(y)):
            return s

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        idx, test_size=cfg.test_size, stratify=y, random_state=cfg.split_seed)
    train_idx, test_idx = np.sort(train_idx), np.sort(test_idx)
    split = {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist(),
             "test_size": cfg.test_size, "random_state": cfg.split_seed, "n_samples": int(len(y))}
    save_json(split, split_path)

    meta = pd.read_csv(fdir / "metadata.csv")
    meta["split"] = "train"
    meta.loc[test_idx, "split"] = "test"
    meta.to_csv(fdir / "metadata.csv", index=False)

    mfcc = np.load(fdir / "X_mfcc_seq.npy", mmap_mode="r")
    mean, std = compute_mfcc_norm_stats(np.asarray(mfcc[train_idx]))
    np.savez(fdir / "mfcc_norm_stats.npz", mean=mean, std=std)
    logger.info("Created split: %d train / %d test songs.", len(train_idx), len(test_idx))
    return split


def build_dataset(cfg: Config, force: bool = False) -> None:
    """Extract features for every song (skipping unreadable files) and cache them."""
    cfg.ensure_dirs()
    if not force and _cache_is_valid(cfg):
        logger.info("Feature cache is up to date; skipping extraction.")
        ensure_split(cfg)
        return

    items = scan_dataset(cfg)
    mfccs, vecs, rows, dropped = [], [], [], []
    label_of = {g: i for i, g in enumerate(cfg.genres)}
    for path, genre in tqdm(items, desc="Extracting features", unit="song"):
        try:
            mfcc, vec = extract_song_features(path, cfg)
        except Exception as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            dropped.append({"path": path.relative_to(cfg.data_dir).as_posix(),
                            "genre": genre, "reason": str(exc)})
            continue
        mfccs.append(mfcc)
        vecs.append(vec)
        rows.append({"song_id": len(rows), "path": path.relative_to(cfg.data_dir).as_posix(),
                     "genre": genre, "label": label_of[genre]})

    if not rows:
        raise RuntimeError("No audio file could be processed. Check the dataset folder.")
    shapes = {m.shape for m in mfccs} | {(v.shape[0],) for v in vecs}
    if len({m.shape for m in mfccs}) != 1 or len({v.shape for v in vecs}) != 1:
        raise RuntimeError(f"Inconsistent feature dimensions: {shapes}")

    meta = pd.DataFrame(rows)
    counts = meta["genre"].value_counts()
    if (counts < 2).any() or len(counts) < len(cfg.genres):
        raise RuntimeError(f"Too few usable songs per genre:\n{counts}")
    if (counts < cfg.classical.get("cv_folds", 5)).any():
        logger.warning("Some genres have fewer songs than cv_folds; CV folds will be reduced.")

    X_mfcc = np.stack(mfccs).astype(np.float32)
    X = np.stack(vecs).astype(np.float32)
    y = meta["label"].to_numpy(dtype=np.int64)
    fdir = cfg.features_dir
    np.save(fdir / "X.npy", X)
    np.save(fdir / "X_mfcc_seq.npy", X_mfcc)
    np.save(fdir / "y.npy", y)
    meta["split"] = "train"
    meta.to_csv(fdir / "metadata.csv", index=False)
    pd.DataFrame(dropped, columns=["path", "genre", "reason"]).to_csv(fdir / "dropped_files.csv", index=False)
    save_json({
        "signature": cfg.extraction_signature(),
        "class_names": list(cfg.genres),
        "feature_names": handcrafted_feature_names(cfg),
        "X_shape": list(X.shape), "X_mfcc_shape": list(X_mfcc.shape),
    }, fdir / "dataset_info.json")
    logger.info("Saved X %s, X_mfcc_seq %s, y %s (%d files dropped).",
                X.shape, X_mfcc.shape, y.shape, len(dropped))
    ensure_split(cfg, force=True)


def load_cached_dataset(cfg: Config) -> dict:
    """Load cached arrays + split and validate their consistency."""
    fdir = cfg.features_dir
    if not _cache_is_valid(cfg):
        raise FileNotFoundError(
            "Feature cache missing or outdated. Run:  python -m src.feature_extraction")
    X = np.load(fdir / "X.npy")
    X_mfcc = np.load(fdir / "X_mfcc_seq.npy")
    y = np.load(fdir / "y.npy")
    meta = pd.read_csv(fdir / "metadata.csv")
    info = load_json(fdir / "dataset_info.json")
    split = ensure_split(cfg)
    stats = np.load(fdir / "mfcc_norm_stats.npz")
    train_idx, test_idx = np.array(split["train_idx"]), np.array(split["test_idx"])

    n = len(y)
    if not (len(X) == len(X_mfcc) == len(meta) == n):
        raise ValueError("Cached arrays have inconsistent lengths; rerun with --force.")
    if X.shape[1] != cfg.n_handcrafted or X_mfcc.shape[1:] != (cfg.n_mfcc, cfg.n_frames):
        raise ValueError(f"Cached feature dims {X.shape}, {X_mfcc.shape} do not match config.")
    if set(train_idx) & set(test_idx) or len(train_idx) + len(test_idx) != n:
        raise ValueError("Split is not a partition of the dataset; rerun with --force.")
    return {"X": X, "X_mfcc": X_mfcc, "y": y, "meta": meta, "class_names": info["class_names"],
            "feature_names": info["feature_names"], "train_idx": train_idx, "test_idx": test_idx,
            "norm_mean": stats["mean"], "norm_std": stats["std"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and cache audio features.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="Recompute even if cache is valid.")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    build_dataset(cfg, force=args.force)


if __name__ == "__main__":
    main()
