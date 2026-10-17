"""Central configuration: paths, column groups and clinical constants."""

from __future__ import annotations

import os
from pathlib import Path

# Works both in the repo (<root>/src/heartrisk/config.py) and in the Docker image (/app/src/...).
PROJECT_ROOT = Path(os.getenv("HEARTRISK_HOME", Path(__file__).resolve().parents[2]))

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "processed.cleveland.data"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "heart_clean.csv"
QUALITY_REPORT_PATH = PROJECT_ROOT / "data" / "processed" / "data_quality.json"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = int(os.getenv("HEARTRISK_SEED", "42"))

# --- Raw UCI Cleveland layout (file has no header row) ---------------------------------
RAW_COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]  # fmt: skip
RAW_TARGET = "num"  # 0 = no disease, 1-4 = disease severity
TARGET = "target"  # binary: 1 = disease present

FEATURE_COLUMNS = [c for c in RAW_COLUMNS if c != RAW_TARGET]

# --- Feature groups used by the preprocessing ColumnTransformer ------------------------
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]

# Engineered by ClinicalFeatureAdder
ENGINEERED_NUMERIC = ["hr_pct_max"]
ENGINEERED_BINARY = ["high_bp", "high_chol", "st_depressed"]

FEATURE_DESCRIPTIONS = {
    "age": "Age (years)",
    "sex": "Sex (1 = male, 0 = female)",
    "cp": "Chest pain type (1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic)",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true)",
    "restecg": "Resting ECG (0 normal, 1 ST-T abnormality, 2 LV hypertrophy)",
    "thalach": "Maximum heart rate achieved (bpm)",
    "exang": "Exercise-induced angina (1 = yes)",
    "oldpeak": "ST depression induced by exercise vs rest",
    "slope": "Slope of peak-exercise ST segment (1 up, 2 flat, 3 down)",
    "ca": "Major vessels coloured by fluoroscopy (0-3)",
    "thal": "Thalassemia (3 normal, 6 fixed defect, 7 reversible defect)",
}

# Risk bands applied to the predicted probability of disease.
RISK_BANDS = ((0.33, "low"), (0.66, "moderate"), (1.01, "high"))
