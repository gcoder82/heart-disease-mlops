"""Thin experiment-tracking facade.

``MlflowTracker`` is the real implementation. ``NullTracker`` lets unit tests and quick
smoke runs execute without an MLflow store, while keeping the training code path identical.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class NullTracker:
    enabled = False

    @contextlib.contextmanager
    def run(self, name: str, tags: dict | None = None, nested: bool = False):
        yield None

    def log_params(self, params: dict) -> None:
        pass

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        pass

    def log_artifact(self, path: Path | str, artifact_dir: str | None = None) -> None:
        pass

    def log_model(self, pipeline, X_example, artifact_path: str = "model") -> None:
        pass

    def set_tags(self, tags: dict) -> None:
        pass


class MlflowTracker(NullTracker):
    enabled = True

    def __init__(self, experiment: str, tracking_uri: str | None = None) -> None:
        import mlflow  # imported lazily so serving images never need it

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        log.info("MLflow tracking uri=%s experiment=%s", mlflow.get_tracking_uri(), experiment)

    @contextlib.contextmanager
    def run(self, name: str, tags: dict | None = None, nested: bool = False):
        with self._mlflow.start_run(run_name=name, nested=nested, tags=tags) as active:
            yield active

    def log_params(self, params: dict) -> None:
        self._mlflow.log_params({k: str(v)[:250] for k, v in params.items()})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Path | str, artifact_dir: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=artifact_dir)

    def set_tags(self, tags: dict) -> None:
        self._mlflow.set_tags(tags)

    def log_model(self, pipeline, X_example, artifact_path: str = "model") -> None:
        import mlflow.sklearn
        from mlflow.models import infer_signature

        example = X_example.head(5)
        signature = infer_signature(example, pipeline.predict_proba(example)[:, 1])
        mlflow.sklearn.log_model(pipeline, artifact_path=artifact_path, signature=signature,
                                 input_example=example)  # fmt: skip


def get_tracker(enabled: bool, experiment: str, tracking_uri: str | None = None) -> NullTracker:
    return MlflowTracker(experiment, tracking_uri) if enabled else NullTracker()
