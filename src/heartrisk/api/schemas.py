"""Request / response contracts. Value ranges follow the UCI Cleveland coding."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

EXAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 2,
    "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3, "ca": 0, "thal": 6,
}  # fmt: skip


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_PATIENT})

    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain: 1 typical, 2 atypical, "
                    "3 non-anginal, 4 asymptomatic")  # fmt: skip
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=30, le=250, description="Max heart rate achieved (bpm)")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression (exercise vs rest)")
    slope: int = Field(..., ge=1, le=3, description="Peak-exercise ST slope (1-3)")
    ca: int | None = Field(None, ge=0, le=3, description="Major vessels coloured (0-3); optional")
    thal: int | None = Field(None, description="3 normal, 6 fixed defect, 7 reversible; optional")

    @field_validator("thal")
    @classmethod
    def _thal_values(cls, v):
        if v is not None and v not in (3, 6, 7):
            raise ValueError("thal must be one of 3, 6, 7")
        return v


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    prediction: int = Field(..., description="1 = heart disease predicted, 0 = not predicted")
    label: str
    confidence: float = Field(..., description="Probability of the predicted class")
    probability_disease: float
    risk_level: str = Field(..., description="low / moderate / high")
    model_version: str
    request_id: str


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=100)


class BatchPrediction(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    risk_level: str


class BatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: list[BatchPrediction]
    model_version: str
    request_id: str


class HealthResponse(BaseModel):
    status: str
