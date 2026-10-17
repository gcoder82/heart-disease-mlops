"""API contract tests. Skipped automatically if FastAPI is not installed."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from heartrisk.api.app import create_app  # noqa: E402


@pytest.fixture(scope="module")
def client(trained_dir):
    app = create_app(str(trained_dir / "model.joblib"))
    with TestClient(app) as c:  # context manager runs the lifespan (loads the model)
        yield c


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready"}


def test_predict_happy_path(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["risk_level"] in {"low", "moderate", "high"}
    assert body["model_version"].startswith("1.0.0-")
    assert body["request_id"] == r.headers["x-request-id"]


def test_request_id_is_propagated(client, sample_patient):
    r = client.post("/predict", json=sample_patient, headers={"x-request-id": "abc-123"})
    assert r.json()["request_id"] == "abc-123"


def test_optional_fields_can_be_omitted(client, sample_patient):
    del sample_patient["ca"], sample_patient["thal"]
    assert client.post("/predict", json=sample_patient).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("age", -5), ("age", 500), ("sex", 2), ("cp", 0), ("cp", 5), ("chol", 10),
    ("thal", 4), ("slope", 9), ("oldpeak", -1), ("age", "old"),
])
def test_invalid_values_are_rejected(client, sample_patient, field, value):
    sample_patient[field] = value
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_missing_required_field_is_rejected(client, sample_patient):
    del sample_patient["age"]
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_unknown_field_is_rejected(client, sample_patient):
    sample_patient["bmi"] = 25
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_batch_endpoint(client, sample_patient):
    payload = {"patients": [sample_patient, dict(sample_patient, age=35, thalach=185)]}
    body = client.post("/predict/batch", json=payload).json()
    assert len(body["predictions"]) == 2


def test_empty_batch_rejected(client):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422


def test_model_info(client):
    body = client.get("/model-info").json()
    assert body["model_name"] and "roc_auc" in " ".join(body["metrics"])


def test_metrics_endpoint_counts_requests_and_predictions(client, sample_patient):
    client.post("/predict", json=sample_patient)
    client.post("/predict", json={"nope": 1})
    text = client.get("/metrics").text
    assert 'heart_api_requests_total{method="POST",path="/predict",status="200"}' in text
    assert 'heart_api_requests_total{method="POST",path="/predict",status="422"}' in text
    assert "heart_api_predictions_total" in text
    assert "heart_api_request_duration_seconds_bucket" in text
    assert "heart_api_model_info" in text


def test_unknown_route_does_not_explode_label_cardinality(client):
    assert client.get("/does/not/exist/123").status_code == 404
    assert 'path="unmatched"' in client.get("/metrics").text


def test_not_ready_without_model(tmp_path):
    app = create_app(str(tmp_path / "missing.joblib"))
    with pytest.raises(FileNotFoundError):
        with TestClient(app):
            pass  # startup must fail fast
