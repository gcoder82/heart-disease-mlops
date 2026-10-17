"""FastAPI application.

Endpoints
    POST /predict        one patient  -> prediction + confidence
    POST /predict/batch  up to 100 patients
    GET  /health         liveness  (process is up)
    GET  /ready          readiness (model is loaded)
    GET  /model-info     metadata of the loaded model
    GET  /metrics        Prometheus exposition format
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from .. import __version__, config
from ..predictor import HeartRiskPredictor
from .logging_config import configure_logging
from .metrics import ApiMetrics
from .schemas import (
    BatchPrediction,
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
)

logger = logging.getLogger("heartrisk.api")


def create_app(model_path: str | None = None) -> FastAPI:
    model_path = model_path or os.getenv("MODEL_PATH", str(config.MODEL_PATH))
    metrics = ApiMetrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configure_logging(os.getenv("LOG_LEVEL", "INFO"), os.getenv("LOG_FILE"))
        # Fail fast: if the model cannot be loaded the container should not become Ready.
        predictor = HeartRiskPredictor(model_path)
        app.state.predictor = predictor
        metrics.model_info.labels(model_name=predictor.model_name,
                                  model_version=predictor.version).set(1)  # fmt: skip
        logger.info("model loaded", extra={"event": "startup", "model_name": predictor.model_name,
                                           "model_version": predictor.version})  # fmt: skip
        yield
        logger.info("shutting down", extra={"event": "shutdown"})

    app = FastAPI(
        title="Heart Disease Risk API",
        version=__version__,
        description="Predicts the presence of heart disease from clinical measurements "
                    "(UCI Cleveland dataset). Educational project - not a medical device.",
        lifespan=lifespan,
    )  # fmt: skip
    app.state.metrics = metrics
    app.state.predictor = None

    # ------------------------------------------------------------------ middleware
    @app.middleware("http")
    async def observe(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        metrics.in_progress.inc()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            logger.exception("unhandled error", extra={"event": "error", "request_id": request_id})
            response = JSONResponse({"detail": "Internal server error",
                                     "request_id": request_id}, status_code=500)  # fmt: skip
        finally:
            metrics.in_progress.dec()

        elapsed = time.perf_counter() - started
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")  # bounded label cardinality
        metrics.requests.labels(request.method, path, str(status)).inc()
        metrics.latency.labels(request.method, path).observe(elapsed)
        response.headers["x-request-id"] = request_id
        if path != "/metrics":  # do not spam logs with Prometheus scrapes
            logger.info("request", extra={
                "event": "request", "request_id": request_id, "method": request.method,
                "path": request.url.path, "status": status,
                "duration_ms": round(elapsed * 1000, 2),
                "client": request.client.host if request.client else None,
            })  # fmt: skip
        return response

    # ------------------------------------------------------------------ helpers
    def get_predictor(request: Request) -> HeartRiskPredictor:
        predictor = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return predictor

    # ------------------------------------------------------------------ routes
    @app.get("/", include_in_schema=False)
    def root():
        return {"service": "heart-disease-risk-api", "version": __version__, "docs": "/docs"}

    @app.get("/health", response_model=HealthResponse)
    def health():
        return HealthResponse(status="ok")

    @app.get("/ready", response_model=HealthResponse)
    def ready(request: Request):
        get_predictor(request)
        return HealthResponse(status="ready")

    @app.get("/model-info")
    def model_info(request: Request):
        p = get_predictor(request)
        meta = p.metadata
        return {
            "model_name": p.model_name,
            "model_version": p.version,
            "trained_at_utc": meta.get("trained_at_utc"),
            "metrics": meta.get("metrics", {}),
            "features": meta.get("features", config.FEATURE_COLUMNS),
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(patient: PatientFeatures, request: Request):
        predictor = get_predictor(request)
        record = patient.model_dump()
        result = predictor.predict([record])[0]
        metrics.observe_prediction(result, age=record["age"])
        logger.info("prediction", extra={
            "event": "prediction", "request_id": request.state.request_id,
            "prediction": result["prediction"],
            "probability_disease": result["probability_disease"],
            "risk_level": result["risk_level"], "model_version": predictor.version,
        })  # fmt: skip
        logger.debug("prediction input", extra={"request_id": request.state.request_id,
                                                "features": record})  # fmt: skip
        return PredictionResponse(**result, model_version=predictor.version,
                                  request_id=request.state.request_id)  # fmt: skip

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request):
        predictor = get_predictor(request)
        records = [p.model_dump() for p in batch.patients]
        results = predictor.predict(records)
        for rec, res in zip(records, results, strict=True):
            metrics.observe_prediction(res, age=rec["age"])
        logger.info("batch prediction", extra={
            "event": "prediction_batch", "request_id": request.state.request_id,
            "size": len(results), "positives": sum(r["prediction"] for r in results),
        })  # fmt: skip
        return BatchResponse(
            predictions=[BatchPrediction(**r) for r in results],
            model_version=predictor.version,
            request_id=request.state.request_id,
        )

    @app.get("/metrics", include_in_schema=False)
    def prometheus_metrics():
        return Response(content=metrics.render(), media_type=metrics.content_type)

    return app


app = create_app()
