"""CNN features -> classical classifiers. Run:  python train_hybrid.py

Writes results/metrics/hybrid_results.csv: the full comparison table
(handcrafted baselines + standalone CNN + all hybrids) and selects the best hybrid by
TRAIN cross-validation score (never by test score) for predict_song.py.
"""
import argparse
import logging

import numpy as np
import pandas as pd

from src.classical_models import run_experiment
from src.config import load_config
from src.evaluation import rows_to_frame
from src.feature_extraction import load_cached_dataset
from src.utils import set_global_seed, setup_logging, save_json

logger = logging.getLogger("train_hybrid")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid CNN-feature + classical models.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force-extract", action="store_true", help="Re-extract CNN features.")
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed, tensorflow=True)
    cfg.ensure_dirs()

    from src.cnn_feature_extractor import FEATURE_KEYS, extract_and_save, load_cnn_features

    data = load_cached_dataset(cfg)
    y, tr, te = data["y"], data["train_idx"], data["test_idx"]
    extract_and_save(cfg, data, force=args.force_extract)
    feats = load_cnn_features(cfg)

    hybrid_rows = []
    for key in FEATURE_KEYS:
        f = feats[key]
        if len(f["train"]) != len(tr) or len(f["test"]) != len(te):
            raise ValueError(f"{key}: feature rows do not match the split. Use --force-extract.")
        logger.info("=== CNN features: %s (%d dims) ===", key, f["train"].shape[1])
        hybrid_rows += run_experiment(key, f["train"], f["test"], y[tr], y[te],
                                      data["class_names"], cfg)
    hybrid_df = rows_to_frame(hybrid_rows)

    # Combine with baselines and the standalone CNN into the final comparison table.
    parts = []
    for name in ("baseline_results.csv", "cnn_results.csv"):
        path = cfg.metrics_dir / name
        if path.is_file():
            parts.append(pd.read_csv(path))
        else:
            logger.warning("%s not found; run the matching training script for a complete table.", name)
    base = parts[0] if parts else None
    if base is not None:
        parts[0] = base[base["Feature Key"] == "handcrafted"]  # table spec: handcrafted rows
    full = pd.concat(parts + [hybrid_df], ignore_index=True)
    full.to_csv(cfg.metrics_dir / "hybrid_results.csv", index=False, float_format="%.4f")

    # Model selection uses train-CV only. Caveat: the CNN has already fit the training songs,
    # so CV on CNN features is optimistic; it is still consistent across all hybrid candidates.
    best = hybrid_df.loc[hybrid_df["CV F1"].idxmax()]
    save_json({"run_id": best["Run ID"], "feature_key": best["Feature Key"],
               "feature_source": best["Feature Source"], "model_key": best["Model Key"],
               "model_label": best["Model"], "cv_f1": float(best["CV F1"]),
               "model_path": f"classical_models/{best['Run ID']}.joblib"},
              cfg.classical_dir / "best_model.json")
    logger.info("Best hybrid by CV F1: %s on %s (cv_f1=%.3f)", best["Model"], best["Feature Source"], best["CV F1"])
    logger.info("Saved %s\n%s", cfg.metrics_dir / "hybrid_results.csv",
                full[["Model", "Feature Source", "Accuracy", "Precision", "Recall", "F1"]]
                .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
