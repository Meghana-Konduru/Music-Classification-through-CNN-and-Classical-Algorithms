"""Metrics, confusion matrices, comparison charts, PCA plots and significance tests."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")  # headless-safe (also fine on Windows)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support, silhouette_score)
from sklearn.preprocessing import StandardScaler

from .config import Config

logger = logging.getLogger(__name__)

RESULT_COLUMNS = ["Model", "Feature Source", "Accuracy", "Precision", "Recall", "F1",
                  "CV F1", "Best Params", "Run ID", "Feature Key", "Model Key"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy plus macro-averaged precision / recall / F1."""
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": p, "Recall": r, "F1": f}


def plot_confusion_matrix(y_true, y_pred, class_names: Sequence[str], out_path: Path, title: str) -> None:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels, class_names, rotation=45, ha="right")
    ax.set_yticks(labels, class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=10)
    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in labels:
        for j in labels:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_predictions(run_id: str, y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Sequence[str], cfg: Config, title: str) -> dict:
    """Compute metrics and save confusion matrix, text report and raw predictions."""
    metrics = compute_metrics(y_true, y_pred)
    plot_confusion_matrix(y_true, y_pred, class_names, cfg.cm_dir / f"{run_id}.png",
                          f"{title}\nacc={metrics['Accuracy']:.3f}  macro-F1={metrics['F1']:.3f}")
    report = classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                                   target_names=list(class_names), zero_division=0)
    rdir = cfg.metrics_dir / "reports"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / f"{run_id}.txt").write_text(report, encoding="utf-8")
    pdir = cfg.metrics_dir / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / f"{run_id}.npy", np.asarray(y_pred))
    return metrics


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[RESULT_COLUMNS]


def plot_history(history: dict, out_dir: Path) -> None:
    """Save accuracy and loss curves (training vs validation)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, label in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history[key], label=f"training {key}")
        ax.plot(history[f"val_{key}"], label=f"validation {key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.set_title(f"CNN {label.lower()}")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"cnn_{key}.png", dpi=150)
        plt.close(fig)


def plot_grouped_bars(df: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Grouped bar chart: x = feature source, bars = model."""
    pivot = df.pivot_table(index="Feature Source", columns="Model", values=metric, sort=False)
    ax = pivot.plot(kind="bar", figsize=(11, 5.5), width=0.8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(metric + (" (macro)" if metric != "Accuracy" else ""))
    ax.set_xlabel("")
    ax.set_title(f"{metric} on the held-out test set")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    ax.legend(title="Model", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pca_projection(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray,
                        y_test: np.ndarray, class_names: Sequence[str], out_path: Path,
                        title: str) -> dict:
    """2-D PCA. Scaler and PCA are fit on TRAIN only; test points are projected.

    Also returns silhouette scores (computed on the full standardised features, with genre
    labels) as a numeric measure of how well genres cluster in this representation.
    """
    scaler = StandardScaler().fit(X_train)
    Xtr, Xte = scaler.transform(X_train), scaler.transform(X_test)
    pca = PCA(n_components=2, random_state=42).fit(Xtr)
    Ztr, Zte = pca.transform(Xtr), pca.transform(Xte)

    def _sil(X, y):
        try:
            return float(silhouette_score(X, y))
        except ValueError:
            return float("nan")

    sil_tr, sil_te = _sil(Xtr, y_train), _sil(Xte, y_test)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7, 6))
    for k, name in enumerate(class_names):
        m_tr, m_te = y_train == k, y_test == k
        ax.scatter(Ztr[m_tr, 0], Ztr[m_tr, 1], s=18, alpha=0.45, color=cmap(k), label=name)
        ax.scatter(Zte[m_te, 0], Zte[m_te, 1], s=45, marker="x", color=cmap(k))
    ev = pca.explained_variance_ratio_.sum()
    ax.set_title(f"{title}\n2-PC variance={ev:.2f} | silhouette train={sil_tr:.2f}, test={sil_te:.2f}",
                 fontsize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(title="circles=train, x=test", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return {"explained_variance_2pc": float(ev), "silhouette_train": sil_tr, "silhouette_test": sil_te}


def mcnemar_exact(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test on paired predictions over the same test set."""
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    only_a = int(np.sum(a_ok & ~b_ok))
    only_b = int(np.sum(~a_ok & b_ok))
    n = only_a + only_b
    p = float(binomtest(min(only_a, only_b), n, 0.5).pvalue) if n > 0 else 1.0
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": p}
