"""Training entry-point:  python -m heartrisk.train [--quick] [--no-mlflow]

For every candidate model: tune (grid search, stratified CV) -> re-score on fresh CV folds ->
evaluate on an untouched hold-out set -> log everything to MLflow. The winner is chosen on
*cross-validated ROC-AUC* (never on the test set) and saved together with a metadata file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from . import __version__, config
from .data import dataframe_hash, load_processed, quality_report, train_test
from .modeling import (
    SELECTION_METRIC,
    candidate_models,
    cross_validated_metrics,
    holdout_metrics,
    tune,
)
from .plots import (
    confusion_matrix_plot,
    cv_comparison_plot,
    feature_importance_plot,
    roc_overlay_plot,
)
from .tracking import NullTracker, get_tracker

log = logging.getLogger("heartrisk.train")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True, cwd=config.PROJECT_ROOT)  # fmt: skip
        return out.stdout.strip()
    except Exception:  # not a git checkout, or git missing
        return "unknown"


def _clean_params(params: dict) -> dict:
    return {k.replace("model__", ""): v for k, v in params.items()}


def _feature_reference(X: pd.DataFrame) -> dict:
    """Training-set feature ranges: a baseline for drift monitoring."""
    desc = X.describe().T
    return {
        col: {"min": float(r["min"]), "max": float(r["max"]), "mean": float(r["mean"]),
              "std": float(r["std"])}
        for col, r in desc.iterrows()
    }  # fmt: skip


def run_training(
    data_path: Path | str = config.PROCESSED_PATH,
    models_dir: Path | str = config.MODELS_DIR,
    reports_dir: Path | str = config.REPORTS_DIR,
    quick: bool = False,
    tracker: NullTracker | None = None,
    test_size: float = 0.2,
    cv_folds: int = 5,
    seed: int = config.RANDOM_STATE,
) -> dict:
    tracker = tracker or NullTracker()
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    fig_dir = reports_dir / "figures" / "model"
    models_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_processed(data_path)
    X_train, X_test, y_train, y_test = train_test(df, test_size=test_size, seed=seed)
    log.info("Train=%d  Test=%d  positive rate=%.3f", len(X_train), len(X_test), y_train.mean())

    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    best_params: dict[str, dict] = {}

    with tracker.run("training-session", tags={"stage": "parent", "project": "heartrisk"}):
        tracker.log_params({
            "dataset_rows": len(df), "dataset_sha256": dataframe_hash(df)[:16],
            "train_rows": len(X_train), "test_rows": len(X_test), "test_size": test_size,
            "cv_folds": cv_folds, "seed": seed, "quick_mode": quick,
            "selection_metric": f"cv_{SELECTION_METRIC}_mean",
            "sklearn_version": sklearn.__version__, "git_commit": _git_commit(),
        })  # fmt: skip

        for name, spec in candidate_models(seed, quick).items():
            log.info("=== %s ===", name)
            with tracker.run(name, tags={"stage": "candidate", "model": name}, nested=True):
                search = tune(spec, X_train, y_train, cv_folds, seed)
                pipe = search.best_estimator_
                cv_res = cross_validated_metrics(pipe, X_train, y_train, cv_folds, seed)
                test_res = holdout_metrics(pipe, X_test, y_test)
                metrics = {"grid_best_cv_roc_auc": float(search.best_score_), **cv_res, **test_res}

                params = _clean_params(search.best_params_)
                tracker.log_params({"model_name": name,
                                    "grid_size": len(search.cv_results_["params"]),
                                    **params})  # fmt: skip
                tracker.log_metrics(metrics)

                cm = confusion_matrix_plot(pipe, X_test, y_test, f"{name} - test set",
                                           fig_dir / f"confusion_{name}.png")  # fmt: skip
                tracker.log_artifact(cm, "plots")
                imp = feature_importance_plot(pipe, name, fig_dir / f"importance_{name}.png")
                tracker.log_artifact(imp, "plots")
                grid_csv = fig_dir / f"grid_results_{name}.csv"
                pd.DataFrame(search.cv_results_).drop(columns=["params"]).to_csv(grid_csv)
                tracker.log_artifact(grid_csv, "tuning")
                tracker.log_model(pipe, X_train)

            results[name], fitted[name], best_params[name] = metrics, pipe, params
            log.info("%s  cv_auc=%.3f  test_auc=%.3f  test_recall=%.3f", name,
                     metrics["cv_roc_auc_mean"], metrics["test_roc_auc"], metrics["test_recall"])

        winner = max(results, key=lambda n: results[n][f"cv_{SELECTION_METRIC}_mean"])
        log.info("Selected model: %s", winner)

        cmp_plot = cv_comparison_plot(results, fig_dir / "cv_model_comparison.png")
        roc_plot = roc_overlay_plot(fitted, X_test, y_test, fig_dir / "roc_overlay_test.png")
        tracker.log_artifact(cmp_plot, "plots")
        tracker.log_artifact(roc_plot, "plots")
        tracker.set_tags({"selected_model": winner})
        tracker.log_params({"selected_model": winner})
        tracker.log_metrics({f"final_{k}": v for k, v in results[winner].items()})

        # ---- persist the winning pipeline + metadata -------------------------------------
        model_path = models_dir / "model.joblib"
        joblib.dump(fitted[winner], model_path)
        model_sha = _file_sha(model_path)
        metadata = {
            "model_name": winner,
            "model_version": f"{__version__}-{model_sha[:8]}",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
            "seed": seed,
            "best_params": best_params[winner],
            "metrics": results[winner],
            "all_models": results,
            "data": quality_report(df),
            "split": {"test_size": test_size, "train_rows": len(X_train),
                      "test_rows": len(X_test)},  # fmt: skip
            "features": config.FEATURE_COLUMNS,
            "feature_reference": _feature_reference(X_train),
            "library_versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "pandas": pd.__version__,
                "numpy": np.__version__,
            },
        }
        meta_path = models_dir / "metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2, default=str))
        pd.DataFrame(results).T.to_csv(reports_dir / "model_comparison.csv")
        tracker.log_artifact(meta_path, "final_model")
        tracker.log_artifact(model_path, "final_model")
        tracker.log_artifact(reports_dir / "model_comparison.csv", "final_model")

    return metadata


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description="Train and track heart-disease classifiers")
    p.add_argument("--data", default=str(config.PROCESSED_PATH))
    p.add_argument("--models-dir", default=str(config.MODELS_DIR))
    p.add_argument("--reports-dir", default=str(config.REPORTS_DIR))
    p.add_argument("--experiment", default="heart-disease-risk")
    p.add_argument("--tracking-uri", default=None, help="default ./mlruns")
    p.add_argument("--no-mlflow", action="store_true", help="skip experiment tracking")
    p.add_argument("--quick", action="store_true", help="tiny grids, for smoke tests")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=config.RANDOM_STATE)
    a = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tracker = get_tracker(not a.no_mlflow, a.experiment, a.tracking_uri)
    meta = run_training(a.data, a.models_dir, a.reports_dir, a.quick, tracker,
                        a.test_size, a.cv_folds, a.seed)  # fmt: skip
    m = meta["metrics"]
    print(f"\nSelected: {meta['model_name']}  (version {meta['model_version']})")
    print(f"  CV ROC-AUC   : {m['cv_roc_auc_mean']:.3f} ± {m['cv_roc_auc_std']:.3f}")
    print(f"  Test ROC-AUC : {m['test_roc_auc']:.3f}   recall={m['test_recall']:.3f}  "
          f"precision={m['test_precision']:.3f}  accuracy={m['test_accuracy']:.3f}")


if __name__ == "__main__":
    main()
