"""Train the dilated CNN on MFCC sequences. Run:  python train_cnn.py"""
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.evaluation import evaluate_predictions, plot_history, rows_to_frame
from src.feature_extraction import load_cached_dataset
from src.preprocessing import prepare_cnn_input
from src.utils import FEATURE_LABELS, set_global_seed, setup_logging

logger = logging.getLogger("train_cnn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the dilated CNN.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed, tensorflow=True)
    cfg.ensure_dirs()

    # Imported after seeding so TF initialises with the seed already set.
    from src.cnn_model import build_dilated_cnn, check_cnn_input, fit_cnn, load_best_cnn

    data = load_cached_dataset(cfg)
    y, train_idx, test_idx = data["y"], data["train_idx"], data["test_idx"]
    # Normalisation stats come from TRAIN songs only (computed in feature_extraction).
    X_cnn = prepare_cnn_input(data["X_mfcc"], data["norm_mean"], data["norm_std"])
    check_cnn_input(X_cnn, cfg)

    # Validation set is carved out of the TRAIN set. Test songs are never used for training.
    tr_idx, val_idx = train_test_split(train_idx, test_size=cfg.cnn["val_size"],
                                       stratify=y[train_idx], random_state=cfg.seed)
    logger.info("CNN data: %d train / %d val / %d test songs", len(tr_idx), len(val_idx), len(test_idx))

    model = build_dilated_cnn(X_cnn.shape[1:], len(data["class_names"]), cfg)
    model.summary(print_fn=logger.info)
    history = fit_cnn(model, X_cnn[tr_idx], y[tr_idx], X_cnn[val_idx], y[val_idx], cfg)
    plot_history(history, cfg.plots_dir)
    logger.info("Best validation accuracy: %.3f", max(history["val_accuracy"]))

    best = load_best_cnn(cfg)   # weights with the lowest validation loss
    probs = best.predict(X_cnn[test_idx], verbose=0)
    y_pred = probs.argmax(axis=1)
    metrics = evaluate_predictions("mfcc_sequence__cnn", y[test_idx], y_pred, data["class_names"],
                                   cfg, "Dilated CNN | MFCC sequence")
    row = {"Model": "CNN", "Feature Source": FEATURE_LABELS["mfcc_sequence"], **metrics,
           "CV F1": float("nan"), "Best Params": "see config.yaml", "Run ID": "mfcc_sequence__cnn",
           "Feature Key": "mfcc_sequence", "Model Key": "cnn"}
    out = cfg.metrics_dir / "cnn_results.csv"
    rows_to_frame([row]).to_csv(out, index=False, float_format="%.4f")
    logger.info("TEST accuracy=%.3f macro-F1=%.3f  (saved %s)", metrics["Accuracy"], metrics["F1"], out)


if __name__ == "__main__":
    main()
