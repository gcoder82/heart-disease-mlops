"""Shared fixtures. Tests use a *synthetic* Cleveland-shaped dataset so that CI never depends on
network access or on the real UCI file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from heartrisk import config


def make_raw_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    """Synthetic frame with the 14 raw Cleveland columns (incl. 'num' 0-4 and a few NaNs)."""
    rng = np.random.default_rng(seed)
    risk = rng.normal(size=n)
    disease = (risk + rng.normal(scale=0.7, size=n) > 0.1).astype(int)
    df = pd.DataFrame({
        "age": np.clip(rng.normal(54 + 4 * disease, 9), 29, 77).round(),
        "sex": rng.binomial(1, 0.55 + 0.25 * disease),
        "cp": np.where(disease == 1, rng.choice([1, 2, 3, 4], n, p=[.1, .1, .1, .7]),
                       rng.choice([1, 2, 3, 4], n, p=[.1, .3, .4, .2])),
        "trestbps": np.clip(rng.normal(130 + 5 * disease, 17), 94, 200).round(),
        "chol": np.clip(rng.normal(245 + 5 * disease, 50), 126, 564).round(),
        "fbs": rng.binomial(1, 0.15, n),
        "restecg": rng.choice([0, 1, 2], n, p=[.5, .02, .48]),
        "thalach": np.clip(rng.normal(158 - 20 * disease, 20), 71, 202).round(),
        "exang": rng.binomial(1, 0.15 + 0.4 * disease),
        "oldpeak": np.clip(rng.exponential(0.6 + 1.0 * disease), 0, 6.2).round(1),
        "slope": np.where(disease == 1, rng.choice([1, 2, 3], n, p=[.2, .7, .1]),
                          rng.choice([1, 2, 3], n, p=[.6, .3, .1])),
        "ca": np.where(disease == 1, rng.choice([0, 1, 2, 3], n, p=[.3, .3, .25, .15]),
                       rng.choice([0, 1, 2, 3], n, p=[.8, .12, .06, .02])).astype(float),
        "thal": np.where(disease == 1, rng.choice([3, 6, 7], n, p=[.2, .1, .7]),
                         rng.choice([3, 6, 7], n, p=[.8, .05, .15])).astype(float),
    })  # fmt: skip
    df["num"] = np.where(disease == 1, rng.integers(1, 5, n), 0)
    df.loc[[3, 17], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return df[config.RAW_COLUMNS]


def write_raw(df: pd.DataFrame, path) -> None:
    """Write in the UCI on-disk format: no header, '?' for missing."""
    df.to_csv(path, header=False, index=False, na_rep="?")


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    return make_raw_frame()


@pytest.fixture(scope="session")
def raw_file(tmp_path_factory, raw_frame):
    path = tmp_path_factory.mktemp("raw") / "processed.cleveland.data"
    write_raw(raw_frame, path)
    return path


@pytest.fixture(scope="session")
def clean_csv(tmp_path_factory, raw_file):
    from heartrisk.data import prepare

    out = tmp_path_factory.mktemp("processed") / "heart_clean.csv"
    prepare(raw_file, out)
    return out


@pytest.fixture(scope="session")
def trained_dir(tmp_path_factory, clean_csv):
    """Runs the real training code (quick grids, no MLflow) once per test session."""
    from heartrisk.train import run_training

    root = tmp_path_factory.mktemp("artifacts")
    run_training(clean_csv, root / "models", root / "reports", quick=True, cv_folds=3)
    return root / "models"


@pytest.fixture()
def sample_patient() -> dict:
    return {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6}  # fmt: skip
