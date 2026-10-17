"""Model-evaluation figures (confusion matrix, ROC overlay, importances, CV comparison)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve  # noqa: E402

from .features import transformed_feature_names  # noqa: E402

PALETTE = {"logistic_regression": "#2a9d8f", "random_forest": "#e76f51",
           "gradient_boosting": "#264653"}  # fmt: skip


def _finish(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def confusion_matrix_plot(pipeline, X, y, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X, y, display_labels=["No disease", "Disease"], cmap="Blues",
        colorbar=False, ax=ax,
    )  # fmt: skip
    ax.set_title(title)
    return _finish(fig, path)


def roc_overlay_plot(fitted: dict, X, y, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for name, pipe in fitted.items():
        fpr, tpr, _ = roc_curve(y, pipe.predict_proba(X)[:, 1])
        area = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=PALETTE.get(name), label=f"{name} (AUC={area:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves - hold-out test set")  # fmt: skip
    ax.legend(loc="lower right", fontsize=8)
    return _finish(fig, path)


def feature_importance_plot(pipeline, name: str, path: Path, top: int = 15) -> Path:
    model = pipeline.named_steps["model"]
    names = transformed_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        values, label = model.feature_importances_, "Impurity importance"
    elif hasattr(model, "coef_"):
        values, label = model.coef_.ravel(), "Coefficient (standardised scale)"
    else:
        raise ValueError(f"No importance available for {name}")
    order = np.argsort(np.abs(values))[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = ["#e76f51" if values[i] < 0 else PALETTE.get(name, "#2a9d8f") for i in order]
    ax.barh([names[i].split("__", 1)[-1] for i in order], values[order], color=colors)
    ax.set(xlabel=label, title=f"Top features - {name}")
    return _finish(fig, path)


def cv_comparison_plot(results: dict[str, dict], path: Path) -> Path:
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    width = 0.8 / max(len(results), 1)
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, (name, res) in enumerate(results.items()):
        means = [res[f"cv_{m}_mean"] for m in metrics]
        stds = [res[f"cv_{m}_std"] for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=name,
               color=PALETTE.get(name))  # fmt: skip
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.set(ylim=(0.5, 1.0), ylabel="Score", title="5-fold cross-validation (mean ± std)")
    ax.legend(fontsize=8)
    return _finish(fig, path)
