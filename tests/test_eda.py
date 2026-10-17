import pytest

pytest.importorskip("seaborn")

from heartrisk.data import load_processed  # noqa: E402
from heartrisk.eda import run_eda  # noqa: E402


def test_eda_generates_all_figures(clean_csv, tmp_path):
    paths = run_eda(load_processed(clean_csv), tmp_path / "figures")
    pngs = sorted(p.name for p in paths if p.suffix == ".png")
    assert len(pngs) == 7
    assert "06_correlation_heatmap.png" in pngs
    assert all(p.stat().st_size > 5_000 for p in paths if p.suffix == ".png")
    assert (tmp_path / "eda_summary.md").read_text().startswith("# EDA summary")
