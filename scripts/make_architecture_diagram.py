#!/usr/bin/env python3
"""Render docs/architecture.png (re-run after changing the architecture)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture.png"
C = {"data": "#e9c46a", "ml": "#2a9d8f", "ops": "#264653", "k8s": "#e76f51", "obs": "#8d6a9f",
     "ci": "#457b9d"}  # fmt: skip

fig, ax = plt.subplots(figsize=(15, 8.6))
ax.set_xlim(0, 150)
ax.set_ylim(0, 86)
ax.axis("off")


def lane(x, y, w, h, title, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                                fc=color, ec=color, alpha=0.10, lw=1.5))  # fmt: skip
    ax.text(x + 1.2, y + h - 2.2, title, fontsize=11, fontweight="bold", color=color, va="top")


def box(x, y, w, h, text, color, fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1",
                                fc="white", ec=color, lw=2))  # fmt: skip
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.35)
    return (x, y, w, h)


def arrow(a, b, color="#444", style="-|>", rad=0.0, side_a="r", side_b="l", ls="-", label=None):
    def pt(b_, side):
        x, y, w, h = b_
        return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "t": (x + w / 2, y + h),
                "b": (x + w / 2, y)}[side]
    p, q = pt(a, side_a), pt(b, side_b)
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, lw=1.6, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))  # fmt: skip
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 1.3, label, fontsize=7.5, ha="center",
                color=color, style="italic")  # fmt: skip


ax.text(75, 83.5, "Heart-Disease Risk · End-to-End MLOps Architecture", ha="center", fontsize=15,
        fontweight="bold")  # fmt: skip

# --- Lane 1: data & model development
lane(2, 52, 146, 28, "1 · Data & model development (local / CI 'train' job)", C["ml"])
uci = box(5, 60, 17, 11, "UCI Heart\nDisease\n(Cleveland)\n303 × 14", C["data"])
prep = box(28, 60, 19, 11, "download_data.py\nheartrisk.data\nclean · de-dup ·\nbinary target", C["ml"])
eda = box(28, 55, 19, 4, "EDA figures + summary", C["ml"], 8)
feat = box(53, 60, 21, 11, "sklearn Pipeline\nClinicalFeatureAdder →\nimpute · scale · one-hot", C["ml"])
tune = box(80, 60, 21, 11, "GridSearchCV (5-fold)\nLogReg · RandForest ·\nGradBoost → select by\nCV ROC-AUC", C["ml"])
mlf = box(107, 64, 18, 8, "MLflow tracking\nparams · metrics ·\nplots · models", C["ops"])
art = box(107, 54, 18, 8, "Model package\nmodel.joblib +\nmetadata.json", C["ops"])
tests = box(130, 59, 16, 12, "pytest (unit)\ndata · features ·\nmodel · predictor ·\nAPI contract", C["ci"])
arrow(uci, prep); arrow(prep, feat); arrow(feat, tune); arrow(tune, mlf, rad=-0.15)
arrow(tune, art, rad=0.15); arrow(prep, eda, side_a="b", side_b="t")
arrow(tests, art, color=C["ci"], ls="--", side_a="l", side_b="r")
ax.text(127.5, 57.2, "gate", fontsize=7.5, ha="center", color=C["ci"], style="italic")

# --- Lane 2: CI/CD
lane(2, 28, 146, 21, "2 · CI/CD  (GitHub Actions)", C["ci"])
j1 = box(5, 33, 20, 10, "lint\nruff check", C["ci"])
j2 = box(31, 33, 22, 10, "test\npytest + coverage\n(artifacts)", C["ci"])
j3 = box(59, 33, 26, 10, "train\ndata → EDA → MLflow\nupload model/mlruns", C["ci"])
j4 = box(91, 33, 28, 10, "docker\nbuild · run · smoke-test\nhelm lint · kustomize", C["ci"])
j5 = box(125, 33, 21, 10, "publish (main)\nimage → GHCR", C["ci"])
for a, b in ((j1, j2), (j2, j3), (j3, j4), (j4, j5)):
    arrow(a, b, color=C["ci"])
ax.text(75, 29.6, "Any failing step stops the workflow → no image is published", ha="center",
        fontsize=8.5, color=C["ci"], style="italic")  # fmt: skip

# --- Lane 3: runtime
lane(2, 2, 146, 23, "3 · Runtime on Kubernetes", C["k8s"])
ax.text(146, 23.0, "Docker Desktop · Minikube · GKE · EKS · AKS", ha="right", va="top", fontsize=8.5, color=C["k8s"], style="italic")
client = box(5, 8, 15, 11, "Client\ncurl · app ·\nSwagger UI", C["data"])
lb = box(26, 8, 19, 11, "Service\n(LoadBalancer)\n/ Ingress", C["k8s"])
pods = box(51, 5, 30, 17, "Deployment · 2-5 replicas (HPA)\n\nFastAPI + uvicorn container\n/predict /health /ready /metrics\nnon-root · probes · limits", C["k8s"])
prom = box(88, 12, 22, 9, "Prometheus\nscrapes /metrics (pods)", C["obs"])
graf = box(116, 12, 28, 9, "Grafana dashboard\nrate · p95 · errors · confidence", C["obs"])
logs = box(88, 4, 22, 6, "JSON request logs\n(stdout → kubectl logs)", C["obs"], 8)
arrow(client, lb); arrow(lb, pods)
arrow(pods, prom, color=C["obs"], label="pull"); arrow(prom, graf, color=C["obs"])
arrow(pods, logs, color=C["obs"], rad=0.1, side_a="r", side_b="l")
arrow(j4, pods, color=C["k8s"], side_a="b", side_b="t", rad=0.12)
ax.text(84, 27.2, "deploy image", fontsize=7.5, ha="center", color=C["k8s"], style="italic")
arrow(art, j3, color=C["ops"], ls="--", side_a="b", side_b="t", rad=0.25)

fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
