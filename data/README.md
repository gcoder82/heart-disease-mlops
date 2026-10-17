# Data

| Path | Content | Created by |
|---|---|---|
| `raw/processed.cleveland.data` | Untouched UCI file (303 rows x 14 columns, `?` = missing) | `python scripts/download_data.py` |
| `processed/heart_clean.csv` | Cleaned data: numeric types, de-duplicated, binary `target`; NaNs kept for in-pipeline imputation | `python -m heartrisk.data prepare` |
| `processed/data_quality.json` | Row count, missing cells, class balance, SHA-256 | same command |

**Source:** UCI Machine Learning Repository - Heart Disease (id 45), Cleveland Clinic Foundation subset.
https://archive.ics.uci.edu/dataset/45/heart+disease  (Detrano et al., 1989; CC BY 4.0)

**First run:** `make data` (or `python scripts/download_data.py --prepare`). Commit both `raw/` and
`processed/` afterwards so the repository is self-contained (the CI download step falls back to the
committed raw file if UCI is unreachable).

**No internet?** Download the zip in a browser, extract `processed.cleveland.data`, then:
`python scripts/download_data.py --file /path/to/processed.cleveland.data --prepare`

## Column dictionary
| Column | Meaning | Values |
|---|---|---|
| age | Age | years |
| sex | Sex | 1 male, 0 female |
| cp | Chest pain type | 1 typical angina, 2 atypical, 3 non-anginal, 4 asymptomatic |
| trestbps | Resting blood pressure | mm Hg |
| chol | Serum cholesterol | mg/dl |
| fbs | Fasting blood sugar > 120 mg/dl | 1 true, 0 false |
| restecg | Resting ECG | 0 normal, 1 ST-T abnormality, 2 LV hypertrophy |
| thalach | Max heart rate achieved | bpm |
| exang | Exercise-induced angina | 1 yes, 0 no |
| oldpeak | ST depression (exercise vs rest) | numeric |
| slope | Slope of peak-exercise ST segment | 1 up, 2 flat, 3 down |
| ca | Major vessels coloured by fluoroscopy | 0-3 (4 missing) |
| thal | Thalassemia | 3 normal, 6 fixed defect, 7 reversible defect (2 missing) |
| target | Derived: any heart disease | 1 if original `num` > 0 |
