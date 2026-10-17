"""Exploratory data analysis: publication-quality figures + a markdown summary.

    python -m heartrisk.eda            # reads data/processed/heart_clean.csv
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from . import config  # noqa: E402
from .data import load_processed  # noqa: E402

log = logging.getLogger("heartrisk.eda")

COLORS = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "No disease", 1: "Disease"}


def _style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.95)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def class_balance(df: pd.DataFrame, out: Path) -> Path:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(4.8, 4))
    bars = ax.bar([LABELS[i] for i in counts.index], counts.values,
                  color=[COLORS[i] for i in counts.index])  # fmt: skip
    for b, v in zip(bars, counts.values, strict=False):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v}\n({v / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=9)  # fmt: skip
    ax.set(ylabel="Patients", title="Class balance", ylim=(0, counts.max() * 1.2))
    return _save(fig, out / "01_class_balance.png")


def missing_values(df: pd.DataFrame, out: Path) -> Path:
    miss = df.isna().sum()
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(miss.index, miss.values, color="#264653")
    ax.set(ylabel="Missing cells", title="Missing values per column")
    ax.tick_params(axis="x", rotation=60)
    for i, v in enumerate(miss.values):
        if v:
            ax.text(i, v + 0.05, str(int(v)), ha="center", va="bottom", fontsize=9)
    return _save(fig, out / "02_missing_values.png")


def numeric_histograms(df: pd.DataFrame, out: Path) -> Path:
    cols = config.NUMERIC_FEATURES
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.ravel(), cols, strict=False):
        for cls in (0, 1):
            sub = df.loc[df[config.TARGET] == cls, col].dropna()
            ax.hist(sub, bins=18, alpha=0.6, color=COLORS[cls], label=LABELS[cls],
                    edgecolor="white")  # fmt: skip
        ax.set(title=config.FEATURE_DESCRIPTIONS[col].split(" (")[0], xlabel=col, ylabel="Count")
    axes[0, 0].legend()
    fig.suptitle("Numeric feature distributions by outcome", y=1.01, fontsize=13)
    return _save(fig, out / "03_numeric_histograms.png")


def boxplots(df: pd.DataFrame, out: Path) -> Path:
    cols = [c for c in config.NUMERIC_FEATURES if c != "ca"]
    fig, axes = plt.subplots(1, len(cols), figsize=(14, 4))
    plot_df = df.assign(outcome=df[config.TARGET].map(LABELS))
    for ax, col in zip(axes, cols, strict=False):
        sns.boxplot(data=plot_df, x="outcome", y=col, hue="outcome", legend=False, ax=ax,
                    palette={LABELS[0]: COLORS[0], LABELS[1]: COLORS[1]})  # fmt: skip
        ax.set(xlabel="", title=col)
    fig.suptitle("Outlier check: numeric features by outcome", y=1.03, fontsize=13)
    return _save(fig, out / "04_boxplots.png")


def categorical_vs_target(df: pd.DataFrame, out: Path) -> Path:
    cols = config.BINARY_FEATURES + config.CATEGORICAL_FEATURES + ["ca"]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.ravel(), cols, strict=False):
        rate = df.groupby(col)[config.TARGET].agg(["mean", "size"])
        ax.bar(rate.index.astype(int).astype(str), rate["mean"], color="#264653")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"], strict=False)):
            ax.text(i, m + 0.01, f"n={n}", ha="center", fontsize=8)
        ax.set(title=col, ylim=(0, 1.1), ylabel="Disease rate", xlabel="")
    for ax in axes.ravel()[len(cols):]:
        ax.axis("off")
    fig.suptitle("Disease rate by categorical / binary feature", y=1.01, fontsize=13)
    return _save(fig, out / "05_categorical_vs_target.png")


def correlation_heatmap(df: pd.DataFrame, out: Path) -> Path:
    corr = df.corr(numeric_only=True)
    mask = None
    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1,
                vmax=1, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"size": 8})  # fmt: skip
    ax.set_title("Correlation matrix (Pearson)")
    return _save(fig, out / "06_correlation_heatmap.png")


def target_correlations(df: pd.DataFrame, out: Path) -> Path:
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET).sort_values()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(corr.index, corr.values, color=["#e76f51" if v > 0 else "#2a9d8f" for v in corr])
    ax.set(xlabel="Pearson r with target", title="Linear association with heart disease")
    return _save(fig, out / "07_target_correlation.png")


def summary_markdown(df: pd.DataFrame, path: Path) -> Path:
    corr = df.corr(numeric_only=True)[config.TARGET].drop(config.TARGET)
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(6)
    lines = [
        "# EDA summary", "",
        f"* Rows: **{len(df)}**, columns: **{df.shape[1]}**",
        f"* Positive (disease) rate: **{df[config.TARGET].mean():.1%}**",
        f"* Missing cells: **{int(df.isna().sum().sum())}** "
        f"({', '.join(f'{c}={int(n)}' for c, n in df.isna().sum().items() if n) or 'none'})",
        "", "## Strongest linear associations with the target", "",
        "| feature | Pearson r |", "|---|---|",
        *[f"| {k} | {v:+.3f} |" for k, v in top.items()],
        "", "## Descriptive statistics", "",
        df.describe().T.round(2).to_markdown() if _has_tabulate() else df.describe().T.round(2)
        .to_string(),
    ]  # fmt: skip
    path.write_text("\n".join(lines))
    return path


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401

        return True
    except ImportError:
        return False


def run_eda(df: pd.DataFrame | None = None, out: Path | str = config.FIGURES_DIR) -> list[Path]:
    _style()
    df = load_processed() if df is None else df
    out = Path(out)
    paths = [
        class_balance(df, out), missing_values(df, out), numeric_histograms(df, out),
        boxplots(df, out), categorical_vs_target(df, out), correlation_heatmap(df, out),
        target_correlations(df, out),
        summary_markdown(df, out.parent / "eda_summary.md"),
    ]  # fmt: skip
    log.info("EDA artefacts written to %s", out)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA figures")
    parser.add_argument("--data", default=str(config.PROCESSED_PATH))
    parser.add_argument("--out", default=str(config.FIGURES_DIR))
    a = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_eda(load_processed(a.data), a.out)


if __name__ == "__main__":
    main()
