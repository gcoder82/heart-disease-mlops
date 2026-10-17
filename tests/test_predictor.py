import pytest

from heartrisk.predictor import HeartRiskPredictor, risk_level


@pytest.fixture(scope="module")
def predictor(trained_dir):
    return HeartRiskPredictor(trained_dir / "model.joblib")


def test_risk_level_bands():
    assert risk_level(0.05) == "low"
    assert risk_level(0.50) == "moderate"
    assert risk_level(0.99) == "high"
    assert risk_level(1.0) == "high"


def test_predict_returns_complete_record(predictor, sample_patient):
    out = predictor.predict([sample_patient])[0]
    assert set(out) == {"prediction", "label", "probability_disease", "confidence", "risk_level"}
    assert out["prediction"] in (0, 1)
    assert 0.0 <= out["probability_disease"] <= 1.0
    assert 0.5 <= out["confidence"] <= 1.0


def test_confidence_is_probability_of_predicted_class(predictor, sample_patient):
    out = predictor.predict([sample_patient])[0]
    p = out["probability_disease"]
    expected = p if out["prediction"] == 1 else 1 - p
    assert out["confidence"] == pytest.approx(expected, abs=1e-3)


def test_missing_optional_values_are_imputed(predictor, sample_patient):
    sample_patient["ca"] = None
    sample_patient["thal"] = None
    assert predictor.predict([sample_patient])[0]["prediction"] in (0, 1)


def test_batch_prediction_matches_single(predictor, sample_patient):
    other = dict(sample_patient, age=40, thalach=180, oldpeak=0.0, exang=0)
    batch = predictor.predict([sample_patient, other])
    assert batch[0] == predictor.predict([sample_patient])[0]
    assert batch[1] == predictor.predict([other])[0]


def test_missing_feature_raises(predictor, sample_patient):
    del sample_patient["chol"]
    with pytest.raises(ValueError, match="chol"):
        predictor.predict([sample_patient])


def test_empty_input_raises(predictor):
    with pytest.raises(ValueError):
        predictor.predict([])


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        HeartRiskPredictor(tmp_path / "absent.joblib")


def test_version_comes_from_metadata(predictor):
    assert predictor.version.startswith("1.0.0-")
    assert predictor.model_name in {"logistic_regression", "random_forest"}
