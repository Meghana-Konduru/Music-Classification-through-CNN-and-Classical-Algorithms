"""Predict the genre of one audio file with every model.
Usage:  python predict_song.py --file path/to/song.wav
"""
import argparse
import logging
import sys

from src.config import load_config
from src.utils import set_global_seed, setup_logging

logger = logging.getLogger("predict_song")


def _fmt_conf(conf) -> str:
    return "n/a" if conf is None else f"{conf * 100:.1f}%"


def _print_block(entries: list) -> None:
    for e in entries:
        print(f"{e['name']}:")
        print(f"Genre: {e['genre'].capitalize()}")
        print(f"Confidence: {_fmt_conf(e['confidence'])}\n")


def print_report(report: dict) -> None:
    """Pretty-print the structured result of src.predict.predict_file."""
    line = "-" * 60
    print(f"\n{line}\nMUSIC GENRE PREDICTION RESULTS\n{line}")
    print("\nClassical Models (handcrafted features)\n")
    _print_block(report["classical"])
    print("CNN Model\n")
    _print_block([report["cnn"]])
    print("Hybrid Models (CNN features -> classical classifier)\n")
    _print_block(report["hybrid"])
    b = report["best"]
    print(f"Best Model Prediction (selected by training cross-validation F1 = {b['cv_f1']:.3f})\n")
    print(f"Model: {b['name']}")
    print(f"Genre: {b['genre'].capitalize()}")
    print(f"Confidence: {_fmt_conf(b['confidence'])}\n{line}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Genre prediction for a single audio file.")
    parser.add_argument("--file", required=True, help="Path to a .wav (or other librosa-readable) file.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    set_global_seed(cfg.seed, tensorflow=True)

    from src.predict import load_inference_assets, predict_file
    from src.preprocessing import AudioLoadError

    try:
        assets = load_inference_assets(cfg)
        report = predict_file(args.file, assets, cfg)
    except (FileNotFoundError, AudioLoadError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
