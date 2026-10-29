"""Load ``config.yaml`` into a typed ``Config`` object and resolve all paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
ALL_GTZAN_GENRES = (
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
)


@dataclass
class Config:
    """All experiment settings. Build it with :func:`load_config`."""

    seed: int
    root: Path
    data_dir: Path
    artifacts_dir: Path
    results_dir: Path
    sample_rate: int
    duration: float
    n_mfcc: int
    n_fft: int
    hop_length: int
    genres: tuple
    test_size: float
    split_seed: int
    classical: dict = field(default_factory=dict)
    cnn: dict = field(default_factory=dict)

    # ---- derived audio geometry -------------------------------------------------
    @property
    def n_samples(self) -> int:
        """Number of samples every song is forced to."""
        return int(round(self.sample_rate * self.duration))

    @property
    def n_frames(self) -> int:
        """MFCC time frames (librosa, center=True): 1 + n_samples // hop."""
        return 1 + self.n_samples // self.hop_length

    @property
    def n_handcrafted(self) -> int:
        """Length of the handcrafted vector: MFCC mean/std, chroma mean/std, 5x(mean,std)."""
        return 2 * self.n_mfcc + 2 * 12 + 10

    # ---- directories -------------------------------------------------------------
    @property
    def features_dir(self) -> Path:
        return self.artifacts_dir / "features"

    @property
    def cnn_features_dir(self) -> Path:
        return self.features_dir / "cnn_features"

    @property
    def cnn_dir(self) -> Path:
        return self.artifacts_dir / "cnn"

    @property
    def classical_dir(self) -> Path:
        return self.artifacts_dir / "classical_models"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    @property
    def cm_dir(self) -> Path:
        return self.results_dir / "confusion_matrices"

    @property
    def metrics_dir(self) -> Path:
        return self.results_dir / "metrics"

    def ensure_dirs(self) -> None:
        for d in (self.features_dir, self.cnn_features_dir, self.cnn_dir, self.classical_dir,
                  self.plots_dir, self.cm_dir, self.metrics_dir,
                  self.metrics_dir / "predictions", self.metrics_dir / "reports"):
            d.mkdir(parents=True, exist_ok=True)

    def extraction_signature(self) -> dict:
        """Settings that invalidate cached features when changed."""
        return {
            "sample_rate": self.sample_rate, "duration": self.duration,
            "n_mfcc": self.n_mfcc, "n_fft": self.n_fft, "hop_length": self.hop_length,
            "genres": list(self.genres),
        }


def _resolve(root: Path, value: Union[str, Path]) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (root / p).resolve()


def load_config(path: Union[str, Path, None] = None) -> Config:
    """Read a YAML config file (default: ``config.yaml`` in the project root)."""
    cfg_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    root = cfg_path.parent
    audio, paths, split = raw["audio"], raw["paths"], raw["split"]
    genres = tuple(raw["genres"])
    unknown = [g for g in genres if g not in ALL_GTZAN_GENRES]
    if not genres or unknown:
        raise ValueError(f"Invalid 'genres' in config (unknown: {unknown}). Valid: {ALL_GTZAN_GENRES}")

    return Config(
        seed=int(raw.get("seed", 42)),
        root=root,
        data_dir=_resolve(root, paths["data_dir"]),
        artifacts_dir=_resolve(root, paths["artifacts_dir"]),
        results_dir=_resolve(root, paths["results_dir"]),
        sample_rate=int(audio["sample_rate"]),
        duration=float(audio["duration"]),
        n_mfcc=int(audio["n_mfcc"]),
        n_fft=int(audio["n_fft"]),
        hop_length=int(audio["hop_length"]),
        genres=genres,
        test_size=float(split["test_size"]),
        split_seed=int(split["random_state"]),
        classical=raw.get("classical", {}),
        cnn=raw.get("cnn", {}),
    )
