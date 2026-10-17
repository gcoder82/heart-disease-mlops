import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from heartrisk import config
from heartrisk.data import clean, split_xy
from heartrisk.features import (
    ClinicalFeatureAdder,
    build_pipeline,
    build_preprocessor,
    transformed_feature_names,
)


@pytest.fixture()
def X(raw_frame):
    return split_xy(clean(raw_frame))[0]


def test_adder_creates_expected_columns(X):
    out = ClinicalFeatureAdder().fit_transform(X)
    for col in config.ENGINEERED_NUMERIC + config.ENGINEERED_BINARY:
        assert col in out.columns
    assert len(out) == len(X)


def test_adder_values_are_correct():
    row = pd.DataFrame([{"age": 60, "trestbps": 150, "chol": 200, "thalach": 112,
                         "oldpeak": 0.0}])  # fmt: skip
    out = ClinicalFeatureAdder().transform(row)
    assert out.loc[0, "hr_pct_max"] == pytest.approx(112 / 160)
    assert out.loc[0, "high_bp"] == 1.0
    assert out.loc[0, "high_chol"] == 0.0
    assert out.loc[0, "st_depressed"] == 0.0


def test_adder_preserves_nan_instead_of_guessing():
    row = pd.DataFrame([{"age": 50, "trestbps": np.nan, "chol": 250, "thalach": 140,
                         "oldpeak": 1.2}])  # fmt: skip
    out = ClinicalFeatureAdder().transform(row)
    assert np.isnan(out.loc[0, "high_bp"])
    assert out.loc[0, "high_chol"] == 1.0


def test_adder_does_not_mutate_input(X):
    before = X.copy()
    ClinicalFeatureAdder().transform(X)
    pd.testing.assert_frame_equal(X, before)


def test_adder_requires_dataframe():
    with pytest.raises(TypeError):
        ClinicalFeatureAdder().transform(np.zeros((3, 3)))


def test_preprocessor_output_is_finite_and_scaled(X):
    pipe = build_pipeline(DummyClassifier())
    Xt = pipe.named_steps["clinical"].transform(X)
    Z = pipe.named_steps["prep"].fit_transform(Xt)
    assert np.isfinite(Z).all(), "imputation must remove every NaN"
    age_idx = list(pipe.named_steps["prep"].get_feature_names_out()).index("num__age")
    assert Z[:, age_idx].mean() == pytest.approx(0, abs=1e-6)
    assert Z[:, age_idx].std() == pytest.approx(1, abs=1e-6)


def test_unseen_category_does_not_crash(X):
    from sklearn.linear_model import LogisticRegression

    y = (X["thalach"] < X["thalach"].median()).astype(int)
    pipe = build_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    odd = X.iloc[:2].copy()
    odd["cp"] = 99.0
    assert pipe.predict_proba(odd).shape == (2, 2)


def test_feature_names_are_prefixed_by_group(X):
    pre = build_preprocessor()
    pre.fit(ClinicalFeatureAdder().transform(X))
    names = list(pre.get_feature_names_out())
    assert any(n.startswith("num__") for n in names)
    assert any(n.startswith("bin__") for n in names)
    assert any(n.startswith("cat__cp_") for n in names)
    assert "num__hr_pct_max" in names


def test_transformed_feature_names_helper(trained_dir):
    import joblib

    pipe = joblib.load(trained_dir / "model.joblib")
    assert len(transformed_feature_names(pipe)) > len(config.FEATURE_COLUMNS)
