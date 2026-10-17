"""Data loading, cleaning and persistence.

Design note: missing values (``ca`` and ``thal`` in the Cleveland file) are *kept* as NaN in the
cleaned CSV and imputed inside the sklearn Pipeline. Imputing here, before the train/test split,
would leak test-set statistics into training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config

log = logging.getLogger(__name__)


def read_raw(path: Path | str = config.RAW_PATH) -> pd.DataFrame:
    """Read the headerless UCI file. '?' marks a missing value."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    df = pd.read_csv(path, header=None, names=config.RAW_COLUMNS, na_values=["?"],
                     skipinitialspace=True)  # fmt: skip
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-coerce, de-duplicate and derive the binary target. Does not impute."""
    missing_cols = [c for c in config.RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data is missing required columns: {missing_cols}")

    out = df[config.RAW_COLUMNS].copy()
    for col in config.RAW_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    before = len(out)
    out = out.dropna(subset=[config.RAW_TARGET])
    out = out.drop_duplicates().reset_index(drop=True)
    if len(out) != before:
        log.info("Dropped %d rows (missing label or duplicates)", before - len(out))

    out[config.TARGET] = (out[config.RAW_TARGET] > 0).astype(int)
    out = out.drop(columns=[config.RAW_TARGET])
    out[config.FEATURE_COLUMNS] = out[config.FEATURE_COLUMNS].astype(float)
    return out


def quality_report(df: pd.DataFrame) -> dict:
    """Small, serialisable summary used by the report and by CI logs."""
    balance = df[config.TARGET].value_counts().sort_index()
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "class_counts": {str(k): int(v) for k, v in balance.items()},
        "positive_rate": round(float(df[config.TARGET].mean()), 4),
        "sha256": dataframe_hash(df),
    }


def dataframe_hash(df: pd.DataFrame) -> str:
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare(raw_path: Path | str = config.RAW_PATH,
            out_path: Path | str = config.PROCESSED_PATH) -> pd.DataFrame:  # fmt: skip
    """raw file -> cleaned CSV (+ data_quality.json next to it)."""
    df = clean(read_raw(raw_path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    report = quality_report(df)
    out_path.with_name("data_quality.json").write_text(json.dumps(report, indent=2))
    log.info("Wrote %s (%d rows). Missing: %s", out_path, len(df), report["missing_by_column"])
    return df


def load_processed(path: Path | str = config.PROCESSED_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m heartrisk.data prepare`.")
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURE_COLUMNS].astype(float), df[config.TARGET].astype(int)


def train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = config.RANDOM_STATE):
    """Stratified hold-out split -> X_train, X_test, y_train, y_test."""
    X, y = split_xy(df)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset utilities")
    parser.add_argument("command", choices=["prepare"])
    parser.add_argument("--raw", default=str(config.RAW_PATH))
    parser.add_argument("--out", default=str(config.PROCESSED_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prepare(args.raw, args.out)


if __name__ == "__main__":
    main()
