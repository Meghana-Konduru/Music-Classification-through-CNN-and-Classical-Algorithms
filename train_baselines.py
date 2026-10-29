"""Train classical models on handcrafted features. Run:  python train_baselines.py"""
import argparse
import logging

from src.classical_models import run_experiment
from src.config import load_config
from src.evaluation import rows_to_frame
from src.feature_extraction import load_cached_dataset
from src.utils import set_global_seed, setup_logging

logger = logging.getLogger("train_baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classical ML baselines.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed)
    cfg.ensure_dirs()

    data = load_cached_dataset(cfg)
    X, y, tr, te = data["X"], data["y"], data["train_idx"], data["test_idx"]
    sources = {
        "handcrafted": X,                          # full vector (MFCC + chroma + spectral + ZCR + RMS)
        "mfcc_summary": X[:, : 2 * cfg.n_mfcc],    # MFCC mean/std only ("raw MFCC" baseline)
    }
    rows = []
    for key, mat in sources.items():
        logger.info("=== Feature source: %s (%d dims) ===", key, mat.shape[1])
        rows += run_experiment(key, mat[tr], mat[te], y[tr], y[te], data["class_names"], cfg)

    df = rows_to_frame(rows)
    out = cfg.metrics_dir / "baseline_results.csv"
    df.to_csv(out, index=False, float_format="%.4f")
    logger.info("Saved %s\n%s", out,
                df[["Model", "Feature Source", "Accuracy", "Precision", "Recall", "F1", "CV F1"]]
                .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
