"""Feature engineering + preprocessing, packaged as sklearn transformers.

Everything that touches data lives *inside* the Pipeline, so the exact same transformation is
applied at training time, in cross-validation, and in the serving API (no train/serve skew).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class ClinicalFeatureAdder(TransformerMixin, BaseEstimator):
    """Adds four clinically motivated features. NaN-safe and stateless.

    hr_pct_max    : achieved max heart rate as a fraction of the age-predicted max (220 - age).
    high_bp       : resting systolic BP >= 140 mm Hg (stage-2 hypertension threshold).
    high_chol     : serum cholesterol >= 240 mg/dl ("high" per AHA).
    st_depressed  : any exercise-induced ST depression (oldpeak > 0).
    """

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else None
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureAdder expects a pandas DataFrame with named columns")
        X = X.copy()
        predicted_max = (220.0 - X["age"]).clip(lower=1.0)
        X["hr_pct_max"] = X["thalach"] / predicted_max
        X["high_bp"] = _flag(X["trestbps"] >= 140, X["trestbps"])
        X["high_chol"] = _flag(X["chol"] >= 240, X["chol"])
        X["st_depressed"] = _flag(X["oldpeak"] > 0, X["oldpeak"])
        return X


def _flag(condition: pd.Series, source: pd.Series) -> pd.Series:
    """Boolean -> float 0/1, keeping NaN where the source value is unknown."""
    return condition.astype(float).where(source.notna(), np.nan)


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    binary = Pipeline([("impute", SimpleImputer(strategy="most_frequent"))])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC_FEATURES + config.ENGINEERED_NUMERIC),
            ("bin", binary, config.BINARY_FEATURES + config.ENGINEERED_BINARY),
            ("cat", categorical, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def build_pipeline(estimator) -> Pipeline:
    """Full model pipeline: raw 13-column DataFrame in -> probability out."""
    return Pipeline([
        ("clinical", ClinicalFeatureAdder()),
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    names = pipeline.named_steps["prep"].get_feature_names_out()
    return [str(n) for n in names]
