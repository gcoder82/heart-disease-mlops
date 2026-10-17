import json

import pandas as pd
import pytest

from heartrisk import config
from heartrisk.data import clean, load_processed, prepare, quality_report, read_raw, train_test


def test_read_raw_parses_question_marks_as_nan(raw_file):
    df = read_raw(raw_file)
    assert list(df.columns) == config.RAW_COLUMNS
    assert df["ca"].isna().sum() == 2
    assert df["thal"].isna().sum() == 1


def test_read_raw_missing_file_has_helpful_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        read_raw(tmp_path / "nope.data")


def test_clean_binarises_target_and_drops_num(raw_frame):
    out = clean(raw_frame)
    assert config.TARGET in out.columns
    assert config.RAW_TARGET not in out.columns
    assert set(out[config.TARGET].unique()) == {0, 1}
    assert (out[config.TARGET] == (raw_frame["num"].loc[out.index] > 0).astype(int)).all()


def test_clean_keeps_nan_for_pipeline_imputation(raw_frame):
    out = clean(raw_frame)
    assert out[["ca", "thal"]].isna().sum().sum() == 3


def test_clean_removes_exact_duplicates(raw_frame):
    doubled = pd.concat([raw_frame, raw_frame.iloc[:10]], ignore_index=True)
    assert len(clean(doubled)) == len(clean(raw_frame))


def test_clean_rejects_missing_columns(raw_frame):
    with pytest.raises(ValueError, match="missing required columns"):
        clean(raw_frame.drop(columns=["chol"]))


def test_clean_coerces_garbage_to_nan(raw_frame):
    bad = raw_frame.astype(object).copy()
    bad.loc[0, "chol"] = "n/a"
    assert pd.isna(clean(bad).loc[0, "chol"])


def test_prepare_writes_csv_and_quality_report(clean_csv):
    df = load_processed(clean_csv)
    report = json.loads(clean_csv.with_name("data_quality.json").read_text())
    assert report["rows"] == len(df)
    assert report["missing_by_column"] == {"ca": 2, "thal": 1}
    assert set(report["class_counts"]) == {"0", "1"}


def test_quality_report_is_deterministic(raw_frame):
    df = clean(raw_frame)
    assert quality_report(df)["sha256"] == quality_report(df.copy())["sha256"]


def test_split_is_stratified_and_reproducible(clean_csv):
    df = load_processed(clean_csv)
    X_tr, X_te, y_tr, y_te = train_test(df, test_size=0.25, seed=1)
    assert len(X_te) == pytest.approx(len(df) * 0.25, abs=1)
    assert abs(y_tr.mean() - y_te.mean()) < 0.05
    X_tr2, *_ = train_test(df, test_size=0.25, seed=1)
    pd.testing.assert_frame_equal(X_tr, X_tr2)


def test_prepare_roundtrip_matches_clean(raw_file, tmp_path):
    out = tmp_path / "x.csv"
    prepare(raw_file, out)
    pd.testing.assert_frame_equal(
        load_processed(out), clean(read_raw(raw_file)), check_dtype=False
    )
