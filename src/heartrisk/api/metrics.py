"""Prometheus metrics, held in a per-app registry (keeps tests isolated)."""

from __future__ import annotations

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


class ApiMetrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "heart_api_requests_total", "HTTP requests handled",
            ["method", "path", "status"], registry=self.registry)  # fmt: skip
        self.latency = Histogram(
            "heart_api_request_duration_seconds", "Request latency in seconds",
            ["method", "path"], registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))  # fmt: skip
        self.in_progress = Gauge(
            "heart_api_requests_in_progress", "Requests currently being processed",
            registry=self.registry)  # fmt: skip
        self.predictions = Counter(
            "heart_api_predictions_total", "Predictions served, by predicted label",
            ["label"], registry=self.registry)  # fmt: skip
        self.confidence = Histogram(
            "heart_api_prediction_confidence", "Confidence of the predicted class",
            registry=self.registry, buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0))  # fmt: skip
        self.probability = Histogram(
            "heart_api_prediction_probability", "Predicted probability of heart disease",
            registry=self.registry, buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
        self.input_age = Histogram(
            "heart_api_input_age_years", "Age of scored patients (input-drift signal)",
            registry=self.registry, buckets=(30, 40, 50, 60, 70, 80, 120))  # fmt: skip
        self.model_info = Gauge(
            "heart_api_model_info", "Loaded model (value is always 1)",
            ["model_name", "model_version"], registry=self.registry)  # fmt: skip

    def observe_prediction(self, result: dict, age: float | None = None) -> None:
        self.predictions.labels(label=result["label"]).inc()
        self.confidence.observe(result["confidence"])
        self.probability.observe(result["probability_disease"])
        if age is not None:
            self.input_age.observe(age)

    def render(self) -> bytes:
        return generate_latest(self.registry)
