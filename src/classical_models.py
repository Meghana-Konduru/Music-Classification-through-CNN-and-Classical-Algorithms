"""Classical classifiers as leakage-safe sklearn Pipelines tuned with train-only CV.

Every feature source (handcrafted, CNN layer 1, ...) goes through the SAME estimators and the
SAME hyper-parameter grids, so only the input features differ between experiments.

Gaussian classifier: ``LinearDiscriminantAnalysis`` (shared covariance) or
``QuadraticDiscriminantAnalysis`` (per-class covariance) is the practical scikit-learn
equivalent of the Gaussian Discriminant Analysis discussed in the reference paper
(class-conditional Gaussians + Bayes rule).
"""
from __future__ import annotations

import logging
import time
from typing import Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (LinearDiscriminantAnalysis,
                                           QuadraticDiscriminantAnalysis)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import Config
from .evaluation import evaluate_predictions
from .utils import FEATURE_LABELS, check_feature_matrix, save_joblib

logger = logging.getLogger(__name__)

MODEL_KEYS = ["logistic_regression", "svm", "random_forest", "gaussian"]


def model_label(key: str, cfg: Config) -> str:
    if key == "gaussian":
        return f"Gaussian ({cfg.classical.get('gaussian', {}).get('kind', 'lda').upper()})"
    return {"logistic_regression": "Logistic Regression", "svm": "SVM",
            "random_forest": "Random Forest"}[key]


def _pca_choices(cfg: Config) -> list:
    """'passthrough' (no PCA) and/or PCA keeping a fraction of the variance."""
    choices: list = []
    for opt in cfg.classical.get("pca_options", [None]):
        if opt in (None, False, "none"):
            if "passthrough" not in choices:
                choices.append("passthrough")
        else:
            choices.append(PCA(n_components=float(opt), svd_solver="full", random_state=cfg.seed))
    return choices


def build_search(key: str, cfg: Config, n_train_per_class_min: int) -> GridSearchCV:
    """Pipeline(scaler -> optional PCA -> classifier) wrapped in a train-only CV grid search."""
    c = cfg.classical
    grid: dict = {"pca": _pca_choices(cfg)}
    if key == "logistic_regression":
        p = c["logistic_regression"]
        clf = LogisticRegression(max_iter=max(2000, int(p.get("max_iter", 3000))))
        grid["clf__C"] = p["C"]
    elif key == "svm":
        p = c["svm"]
        clf = SVC(kernel="rbf", probability=True, random_state=cfg.seed)
        grid["clf__C"], grid["clf__gamma"] = p["C"], p["gamma"]
    elif key == "random_forest":
        p = c["random_forest"]
        clf = RandomForestClassifier(random_state=cfg.seed, n_jobs=1)
        grid["clf__n_estimators"], grid["clf__max_depth"] = p["n_estimators"], p["max_depth"]
    elif key == "gaussian":
        p = c["gaussian"]
        if p.get("kind", "lda").lower() == "qda":
            clf = QuadraticDiscriminantAnalysis()
            grid["clf__reg_param"] = p["qda_reg_param"]
        else:
            clf = LinearDiscriminantAnalysis(solver="lsqr")
            grid["clf__shrinkage"] = p["lda_shrinkage"]
    else:
        raise ValueError(f"Unknown model key: {key}")

    pipe = Pipeline([("scaler", StandardScaler()), ("pca", "passthrough"), ("clf", clf)])
    folds = max(2, min(int(c.get("cv_folds", 5)), n_train_per_class_min))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
    return GridSearchCV(pipe, grid, cv=cv, scoring=c.get("scoring", "f1_macro"),
                        n_jobs=-1, refit=True, error_score="raise")


def _describe_params(params: dict) -> str:
    out = []
    for k, v in params.items():
        if k == "pca":
            v = "none" if v == "passthrough" else f"PCA({v.n_components})"
        out.append(f"{k.replace('clf__', '')}={v}")
    return ", ".join(out)


def run_experiment(feature_key: str, X_train: np.ndarray, X_test: np.ndarray,
                   y_train: np.ndarray, y_test: np.ndarray, class_names: Sequence[str],
                   cfg: Config, model_keys: Sequence[str] = MODEL_KEYS) -> list[dict]:
    """Tune, fit (on train only) and evaluate each classical model on one feature source."""
    check_feature_matrix(X_train, f"{feature_key} train")
    check_feature_matrix(X_test, f"{feature_key} test")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"{feature_key}: train/test feature dimensions differ "
                         f"({X_train.shape[1]} vs {X_test.shape[1]})")
    min_per_class = int(np.bincount(y_train).min())
    rows = []
    for mk in model_keys:
        t0 = time.time()
        search = build_search(mk, cfg, min_per_class)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        y_pred = model.predict(X_test)

        run_id = f"{feature_key}__{mk}"
        label = model_label(mk, cfg)
        source = FEATURE_LABELS[feature_key]
        metrics = evaluate_predictions(run_id, y_test, y_pred, class_names, cfg, f"{label} | {source}")
        save_joblib(model, cfg.classical_dir / f"{run_id}.joblib")
        row = {"Model": label, "Feature Source": source, **metrics,
               "CV F1": float(search.best_score_), "Best Params": _describe_params(search.best_params_),
               "Run ID": run_id, "Feature Key": feature_key, "Model Key": mk}
        rows.append(row)
        logger.info("%-22s | %-20s | cv_f1=%.3f test_acc=%.3f test_f1=%.3f (%.0fs)",
                    label, source, row["CV F1"], metrics["Accuracy"], metrics["F1"], time.time() - t0)
    return rows
