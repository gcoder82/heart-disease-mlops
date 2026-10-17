"""Framework-free inference wrapper (no FastAPI / MLflow imports -> trivially unit-testable)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from . import config


def risk_level(probability: float) -> str:
    for upper, label in config.RISK_BANDS:
        if probability < upper:
            return label
    return config.RISK_BANDS[-1][1]


class HeartRiskPredictor:
    """Loads the persisted sklearn Pipeline (+ metadata) and scores patient records."""

    def __init__(self, model_path: Path | str = config.MODEL_PATH,
                 metadata_path: Path | str | None = None) -> None:  # fmt: skip
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.pipeline = joblib.load(model_path)
        meta_path = Path(metadata_path) if metadata_path else model_path.with_name("metadata.json")
        self.metadata: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.model_name: str = self.metadata.get("model_name", "unknown")
        self.version: str = self.metadata.get("model_version", "unversioned")

    def predict(self, records: list[dict]) -> list[dict]:
        if not records:
            raise ValueError("No records supplied")
        frame = pd.DataFrame(records)
        missing = [c for c in config.FEATURE_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")
        frame = frame[config.FEATURE_COLUMNS].astype(float)  # None -> NaN, imputed in pipeline

        proba = self.pipeline.predict_proba(frame)[:, 1]
        results = []
        for p in proba:
            p = float(p)
            label = int(p >= 0.5)
            results.append({
                "prediction": label,
                "label": "heart_disease" if label else "no_heart_disease",
                "probability_disease": round(p, 4),
                "confidence": round(max(p, 1.0 - p), 4),
                "risk_level": risk_level(p),
            })
        return results
