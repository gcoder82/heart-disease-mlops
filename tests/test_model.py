import json

import joblib
import pytest

from heartrisk import config
from heartrisk.data import load_processed, train_test
from heartrisk.modeling import METRIC_NAMES, candidate_models, holdout_metrics
from heartrisk.tracking import NullTracker


def test_candidate_models_cover_at_least_two_algorithms():
    assert len(candidate_models(0, quick=False)) >= 2
    assert {"logistic_regression", "random_forest"} <= set(candidate_models(0, quick=True))


def test_training_writes_all_artifacts(trained_dir):
    assert (trained_dir / "model.joblib").exists()
    meta = json.loads((trained_dir / "metadata.json").read_text())
    assert meta["model_name"] in {"logistic_regression", "random_forest"}
    assert meta["features"] == config.FEATURE_COLUMNS
    assert set(meta["all_models"]) >= {"logistic_regression", "random_forest"}
    assert meta["model_version"].startswith("1.0.0-")


def test_metrics_are_valid_probabilities(trained_dir):
    meta = json.loads((trained_dir / "metadata.json").read_text())
    for model_metrics in meta["all_models"].values():
        for m in METRIC_NAMES:
            assert 0.0 <= model_metrics[f"cv_{m}_mean"] <= 1.0
            assert model_metrics[f"cv_{m}_std"] >= 0.0
        assert 0.0 <= model_metrics["test_roc_auc"] <= 1.0


def test_model_beats_chance_on_signal_bearing_data(trained_dir):
    meta = json.loads((trained_dir / "metadata.json").read_text())
    assert meta["metrics"]["cv_roc_auc_mean"] > 0.75


def test_winner_is_chosen_by_cv_auc_not_test(trained_dir):
    meta = json.loads((trained_dir / "metadata.json").read_text())
    best = max(meta["all_models"], key=lambda n: meta["all_models"][n]["cv_roc_auc_mean"])
    assert meta["model_name"] == best


def test_saved_pipeline_scores_raw_dataframe(trained_dir, clean_csv):
    pipe = joblib.load(trained_dir / "model.joblib")
    X = load_processed(clean_csv)[config.FEATURE_COLUMNS].head(5)
    proba = pipe.predict_proba(X)
    assert proba.shape == (5, 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_saved_pipeline_reproduces_holdout_metrics(trained_dir, clean_csv):
    pipe = joblib.load(trained_dir / "model.joblib")
    meta = json.loads((trained_dir / "metadata.json").read_text())
    _, X_te, _, y_te = train_test(load_processed(clean_csv), 0.2, meta["seed"])
    again = holdout_metrics(pipe, X_te, y_te)
    assert again["test_roc_auc"] == pytest.approx(meta["metrics"]["test_roc_auc"])


def test_training_is_deterministic(clean_csv, tmp_path):
    from heartrisk.train import run_training

    a = run_training(clean_csv, tmp_path / "a", tmp_path / "ra", quick=True, cv_folds=3, seed=7)
    b = run_training(clean_csv, tmp_path / "b", tmp_path / "rb", quick=True, cv_folds=3, seed=7)
    assert a["metrics"]["test_roc_auc"] == pytest.approx(b["metrics"]["test_roc_auc"])
    assert a["best_params"] == b["best_params"]


def test_training_figures_are_produced(trained_dir):
    fig_dir = trained_dir.parent / "reports" / "figures" / "model"
    names = {p.name for p in fig_dir.glob("*.png")}
    assert {"roc_overlay_test.png", "cv_model_comparison.png"} <= names
    assert any(n.startswith("confusion_") for n in names)


def test_null_tracker_is_inert():
    t = NullTracker()
    with t.run("x", tags={"a": "b"}, nested=True):
        t.log_params({"a": 1})
        t.log_metrics({"m": 1.0})
    assert t.enabled is False


def test_mlflow_tracker_logs_params_metrics_and_artifacts(clean_csv, tmp_path):
    pytest.importorskip("mlflow")
    import mlflow

    from heartrisk.tracking import MlflowTracker
    from heartrisk.train import run_training

    uri = (tmp_path / "mlruns").as_uri()
    tracker = MlflowTracker("pytest-experiment", uri)
    run_training(clean_csv, tmp_path / "m", tmp_path / "r", quick=True, cv_folds=3,
                 tracker=tracker)  # fmt: skip

    client = mlflow.tracking.MlflowClient(uri)
    exp = client.get_experiment_by_name("pytest-experiment")
    runs = client.search_runs([exp.experiment_id])
    assert len(runs) == 3  # 1 parent + 2 candidate runs
    candidates = [r for r in runs if r.data.tags.get("stage") == "candidate"]
    assert len(candidates) == 2
    for r in candidates:
        assert "cv_roc_auc_mean" in r.data.metrics
        assert "test_recall" in r.data.metrics
        assert "model_name" in r.data.params
        artifacts = {a.path for a in client.list_artifacts(r.info.run_id)}
        assert {"plots", "model"} <= artifacts
