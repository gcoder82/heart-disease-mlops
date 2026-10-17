"""Candidate models, hyper-parameter grids and metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from .features import build_pipeline

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}
METRIC_NAMES = list(CV_SCORING)
SELECTION_METRIC = "roc_auc"


def candidate_models(seed: int, quick: bool = False) -> dict[str, tuple[object, dict]]:
    """name -> (estimator, param_grid). ``quick`` shrinks the grids for tests / smoke runs."""
    if quick:
        return {
            "logistic_regression": (
                LogisticRegression(max_iter=2000, random_state=seed),
                {"model__C": [0.1, 1.0]},
            ),
            "random_forest": (
                RandomForestClassifier(random_state=seed, n_jobs=1),
                {"model__n_estimators": [50], "model__max_depth": [3, None]},
            ),
        }
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=seed),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
             "model__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=seed, n_jobs=1),
            {"model__n_estimators": [300],
             "model__max_depth": [None, 4, 8],
             "model__min_samples_leaf": [1, 3, 5],
             "model__max_features": ["sqrt", 0.5]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=seed),
            {"model__n_estimators": [100, 200],
             "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3],
             "model__subsample": [0.8, 1.0]},
        ),
    }  # fmt: skip


def tune(name_estimator_grid, X: pd.DataFrame, y: pd.Series, cv_folds: int, seed: int,
         n_jobs: int = -1) -> GridSearchCV:  # fmt: skip
    """Grid-search the full pipeline (preprocessing is re-fit inside every fold -> no leakage)."""
    estimator, grid = name_estimator_grid
    search = GridSearchCV(
        build_pipeline(clone(estimator)),
        param_grid=grid,
        scoring=SELECTION_METRIC,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        n_jobs=n_jobs,
        refit=True,
    )
    return search.fit(X, y)


def cross_validated_metrics(pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int,
                            seed: int) -> dict[str, float]:  # fmt: skip
    """Mean/std of every metric over a *different* CV shuffle than the one used for tuning.

    GridSearchCV.best_score_ is optimistically biased (it is the max over many configs);
    re-scoring the winner on fresh folds gives a fairer estimate.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed + 1)
    res = cross_validate(clone(pipeline), X, y, cv=cv, scoring=CV_SCORING, n_jobs=1)
    out: dict[str, float] = {}
    for m in METRIC_NAMES:
        scores = res[f"test_{m}"]
        out[f"cv_{m}_mean"] = float(np.mean(scores))
        out[f"cv_{m}_std"] = float(np.std(scores))
    return out


def holdout_metrics(pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": float(accuracy_score(y, pred)),
        "test_precision": float(precision_score(y, pred, zero_division=0)),
        "test_recall": float(recall_score(y, pred, zero_division=0)),
        "test_f1": float(f1_score(y, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y, proba)),
    }
