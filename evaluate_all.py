"""Final comparison charts, PCA plots and significance tests. Run:  python evaluate_all.py"""
import argparse
import logging

import numpy as np
import pandas as pd

from src.cnn_feature_extractor import FEATURE_KEYS, load_cnn_features
from src.config import load_config
from src.evaluation import mcnemar_exact, plot_grouped_bars, plot_pca_projection
from src.feature_extraction import load_cached_dataset
from src.utils import FEATURE_LABELS, set_global_seed, setup_logging

logger = logging.getLogger("evaluate_all")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare all models.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    cfg.ensure_dirs()

    results_path = cfg.metrics_dir / "hybrid_results.csv"
    if not results_path.is_file():
        raise FileNotFoundError("Run train_baselines.py, train_cnn.py and train_hybrid.py first.")
    df = pd.read_csv(results_path)
    df.to_csv(cfg.metrics_dir / "final_comparison.csv", index=False, float_format="%.4f")

    # 1-2. Bar charts
    plot_grouped_bars(df, "Accuracy", cfg.plots_dir / "accuracy_comparison.png")
    plot_grouped_bars(df, "F1", cfg.plots_dir / "f1_comparison.png")

    # 3. Confusion matrices were written per model during training (results/confusion_matrices/).

    # 4-7. PCA plots (scaler + PCA fit on train only) and silhouette scores
    data = load_cached_dataset(cfg)
    y, tr, te, names = data["y"], data["train_idx"], data["test_idx"], data["class_names"]
    cnn_feats = load_cnn_features(cfg)
    reps = {"handcrafted": (data["X"][tr], data["X"][te])}
    reps.update({k: (cnn_feats[k]["train"], cnn_feats[k]["test"]) for k in FEATURE_KEYS})
    quality = []
    for key, (a, b) in reps.items():
        stats = plot_pca_projection(a, b, y[tr], y[te], names, cfg.plots_dir / f"pca_{key}.png",
                                    f"PCA: {FEATURE_LABELS[key]}")
        quality.append({"Feature Key": key, "Feature Source": FEATURE_LABELS[key], "Dims": a.shape[1], **stats})
    qdf = pd.DataFrame(quality)
    qdf.to_csv(cfg.metrics_dir / "representation_quality.csv", index=False, float_format="%.4f")

    # Paired McNemar test: same classifier, handcrafted vs CNN features, same test songs.
    pred_dir = cfg.metrics_dir / "predictions"
    y_test, tests = y[te], []
    for mk in df.loc[df["Feature Key"].isin(FEATURE_KEYS), "Model Key"].unique():
        ref_file = pred_dir / f"handcrafted__{mk}.npy"
        if not ref_file.is_file():
            continue
        ref = np.load(ref_file)
        for fk in FEATURE_KEYS:
            f = pred_dir / f"{fk}__{mk}.npy"
            if f.is_file():
                r = mcnemar_exact(y_test, ref, np.load(f))
                tests.append({"Model Key": mk, "Feature Key": fk,
                              "handcrafted_only_correct": r["only_a_correct"],
                              "cnn_features_only_correct": r["only_b_correct"], "p_value": r["p_value"]})
    pd.DataFrame(tests).to_csv(cfg.metrics_dir / "mcnemar_vs_handcrafted.csv", index=False, float_format="%.4f")

    # Answer to the experimental question
    summary = (df.groupby(["Feature Key", "Feature Source"], sort=False)[["Accuracy", "F1"]]
               .agg(["mean", "max"]).round(4))
    summary.to_csv(cfg.metrics_dir / "summary_by_feature_source.csv")
    logger.info("Mean / max across classifiers, per feature source:\n%s", summary.to_string())
    logger.info("Representation quality (silhouette on standardised features):\n%s",
                qdf[["Feature Source", "Dims", "silhouette_train", "silhouette_test"]].to_string(index=False))
    logger.info("With only %d test songs, 1 song = %.1f%% accuracy; check mcnemar_vs_handcrafted.csv "
                "before claiming a real difference.", len(te), 100 / len(te))
    logger.info("All outputs saved under %s", cfg.results_dir)


if __name__ == "__main__":
    main()
