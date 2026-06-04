"""Smoke tests for Phase E analysis plots and tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

from causal_meta.analysis.generalisation.plots import generate_shift_figure
from causal_meta.analysis.generalisation.tables import \
    generate_distance_regression_table
from causal_meta.analysis.plots.results import (
    _ood_category, _pivot_metrics, generate_calibration_scatter,
    generate_density_stratified_figure, generate_distance_degradation_scatter)


def _make_synthetic_df() -> pd.DataFrame:
    """Build a synthetic DataFrame mimicking load_runs_dataframe() output."""
    rows = []
    models = ["AviCi", "BCNP", "DiBS"]
    datasets = {
        "id_linear_er20": ("er", "linear", 0.0526, 0.0, 0.0),
        "id_linear_er40": ("er", "linear", 0.1053, 0.0, 0.0),
        "id_linear_er60": ("er", "linear", 0.1579, 0.0, 0.0),
        "ood_mech_periodic_er40": ("er", "periodic", 0.1053, 0.8, 1.5),
        "ood_graph_sbm_linear": ("sbm", "linear", None, 1.2, 2.1),
        "ood_both_sbm_periodic": ("sbm", "periodic", None, 1.5, 3.0),
    }
    metrics = [
        "e-shd",
        "e-sid",
        "ne-shd",
        "ne-sid",
        "edge_entropy",
        "auc",
        "graph_nll_per_edge",
    ]

    for model in models:
        for dk, (gt, mt, sp, sd, kl) in datasets.items():
            for metric in metrics:
                rows.append(
                    {
                        "RunID": f"run_{model.lower()}",
                        "RunName": "rq1",
                        "RunDir": "/tmp/fake",
                        "Model": model,
                        "Dataset": dk,
                        "ModelKey": model.lower(),
                        "DatasetKey": dk,
                        "Metric": metric,
                        "Mean": 10.0 + hash((model, dk, metric)) % 100 / 10.0,
                        "SEM": 0.5,
                        "Std": 1.0,
                        "GraphType": gt,
                        "MechType": mt,
                        "NNodes": 20,
                        "SparsityParam": sp,
                        "SpectralDist": sd,
                        "KLDegreeDist": kl,
                    }
                )
    return pd.DataFrame(rows)


# ── OOD category helper ─────────────────────────────────────────────────


class TestOODCategory:
    def test_id(self) -> None:
        assert _ood_category("id_linear_er20") == "ID"

    def test_ood_graph(self) -> None:
        assert _ood_category("ood_graph_sbm_linear") == "OOD-Graph"

    def test_ood_mech(self) -> None:
        assert _ood_category("ood_mech_periodic_er40") == "OOD-Mech"

    def test_ood_both(self) -> None:
        assert _ood_category("ood_both_sbm_periodic") == "OOD-Both"

    def test_ood_noise(self) -> None:
        assert _ood_category("ood_noise_laplace_linear_er20") == "OOD-Noise"

    def test_ood_nodes(self) -> None:
        assert _ood_category("ood_nodes_linear_er20_d40_n500") == "OOD-Nodes"

    def test_ood_samples(self) -> None:
        assert _ood_category("ood_samples_linear_er20_d20_n1000") == "OOD-Samples"


# ── _pivot_metrics ───────────────────────────────────────────────────────


class TestPivotMetrics:
    def test_pivot_produces_columns(self) -> None:
        df = _make_synthetic_df()
        wide = _pivot_metrics(df, ["ne-shd", "edge_entropy"])
        assert not wide.empty
        assert "ne-shd" in wide.columns
        assert "edge_entropy" in wide.columns

    def test_pivot_empty_on_missing_metric(self) -> None:
        df = _make_synthetic_df()
        wide = _pivot_metrics(df, ["nonexistent_metric"])
        assert wide.empty


# ── Calibration scatter (E.2) ────────────────────────────────────────────


def test_calibration_scatter_creates_file(tmp_path: Path) -> None:
    df = _make_synthetic_df()
    out = tmp_path / "cal.png"
    generate_calibration_scatter(df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_calibration_scatter_handles_empty_df(tmp_path: Path) -> None:
    out = tmp_path / "cal.png"
    generate_calibration_scatter(pd.DataFrame(), out)
    assert not out.exists()


# ── Distance-degradation scatter (E.4) ──────────────────────────────────


def test_distance_degradation_scatter_creates_file(tmp_path: Path) -> None:
    df = _make_synthetic_df()
    out = tmp_path / "dist_deg.png"
    generate_distance_degradation_scatter(df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_distance_degradation_scatter_skips_id_only(tmp_path: Path) -> None:
    df = _make_synthetic_df()
    id_only = cast(
        pd.DataFrame,
        df[df["DatasetKey"].astype(str).str.startswith("id_")].copy(),
    )
    out = tmp_path / "dist_deg_id_only.png"
    generate_distance_degradation_scatter(id_only, out)
    assert not out.exists()


# ── Density-stratified plot (E.6) ───────────────────────────────────────


def test_density_stratified_creates_file(tmp_path: Path) -> None:
    df = _make_synthetic_df()
    out = tmp_path / "density.png"
    generate_density_stratified_figure(df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_compound_shift_figure_uses_multi_panel_layout(tmp_path: Path) -> None:
    rows = []
    models = ["AviCi", "BCNP"]
    dataset_keys = [
        "id_linear_er20_d20_n500",
        "id_neuralnet_sf2_d20_n500",
        "id_gpcde_er60_d20_n500",
        "ood_both_sbm_periodic_d20_n500",
        "ood_both_sbm_logistic_map_d20_n500",
        "ood_both_ws_periodic_d20_n500",
        "ood_both_ws_pnl_tanh_d20_n500",
        "ood_both_grg_logistic_map_d20_n500",
        "ood_both_grg_pnl_tanh_d20_n500",
    ]

    for model in models:
        for dataset_key in dataset_keys:
            rows.append(
                {
                    "Model": model,
                    "DatasetKey": dataset_key,
                    "Dataset": dataset_key,
                    "AxisCategory": (
                        "id" if dataset_key.startswith("id_") else "compound"
                    ),
                    "Metric": "ne-sid",
                    "Value": 1.0 + hash((model, dataset_key)) % 10 / 10.0,
                    "NNodes": 20,
                    "SamplesPerTask": 500,
                }
            )

    df = pd.DataFrame(rows)
    out = tmp_path / "compound_shift.pdf"
    agg = generate_shift_figure(df, shift_axis="compound", output_path=out)

    assert out.exists()
    assert out.stat().st_size > 0
    assert not agg.empty
    assert set(agg["FixedConcept"].dropna().unique()) == {"SBM", "WS", "GRG"}


# ── Distance regression table (E.5) ─────────────────────────────────────


def test_distance_regression_table_creates_file(tmp_path: Path) -> None:
    df = _make_synthetic_df()
    out = tmp_path / "reg.tex"
    generate_distance_regression_table(df, out)
    assert out.exists()
    content = out.read_text()
    assert r"\begin{tabular}" in content
    assert "R^2" in content


def test_distance_regression_table_skips_id_only(tmp_path: Path) -> None:
    df = _make_synthetic_df()
    id_only = cast(
        pd.DataFrame,
        df[df["DatasetKey"].astype(str).str.startswith("id_")].copy(),
    )
    out = tmp_path / "reg_id_only.tex"
    generate_distance_regression_table(id_only, out)
    assert not out.exists()


def test_distance_regression_table_handles_empty(tmp_path: Path) -> None:
    out = tmp_path / "reg.tex"
    generate_distance_regression_table(pd.DataFrame(), out)
    assert not out.exists()
