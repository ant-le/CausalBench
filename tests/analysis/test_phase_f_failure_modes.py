"""Tests for failure mode classification and raw data loading."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from causal_meta.analysis.diagnostics.failure_modes import (
    FAILURE_MODE_CATEGORIES,
    _pivot_raw_wide,
    classify_failure_modes,
    failure_mode_fractions,
    ood_category,
)
from causal_meta.analysis.utils import load_raw_task_dataframe


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_raw_task_df(
    sparsity_ratios: list[float],
    skeleton_f1s: list[float],
    orientation_accs: list[float],
    model: str = "TestModel",
    dataset_key: str = "id_linear_er20",
) -> pd.DataFrame:
    """Build a synthetic raw-task long-format DataFrame."""
    rows: list[dict[str, object]] = []
    n_tasks = len(sparsity_ratios)
    assert len(skeleton_f1s) == n_tasks
    assert len(orientation_accs) == n_tasks

    for task_idx in range(n_tasks):
        for metric, vals in [
            ("sparsity_ratio", sparsity_ratios),
            ("skeleton_f1", skeleton_f1s),
            ("orientation_accuracy", orientation_accs),
        ]:
            rows.append(
                {
                    "RunID": "run1",
                    "Model": model,
                    "ModelKey": model.lower(),
                    "DatasetKey": dataset_key,
                    "Dataset": dataset_key,
                    "TaskIdx": task_idx,
                    "Metric": metric,
                    "Value": vals[task_idx],
                    "GraphType": "er",
                    "MechType": "linear",
                    "NNodes": 20,
                    "SparsityParam": 0.1053,
                    "SpectralDist": 0.0,
                    "KLDegreeDist": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _make_metrics_json(
    run_dir: Path,
    model_name: str = "test_model",
    raw_data: dict[str, dict[str, list[float]]] | None = None,
) -> None:
    """Write a synthetic metrics.json into run_dir."""
    if raw_data is None:
        raw_data = {
            "id_linear_er20": {
                "e-shd": [10.0, 12.0, 11.0],
                "sparsity_ratio": [0.02, 1.5, 0.8],
                "skeleton_f1": [0.3, 0.7, 0.9],
                "orientation_accuracy": [0.1, 0.6, 0.8],
            }
        }

    payload = {
        "metadata": {
            "run_id": run_dir.name,
            "run_name": "test",
            "model_name": model_name,
            "output_dir": str(run_dir),
        },
        "family_metadata": {
            "id_linear_er20": {
                "n_nodes": 20,
                "graph_type": "er",
                "mech_type": "linear",
                "sparsity_param": 0.1053,
            }
        },
        "distances": {"id_linear_er20": {"spectral": 0.0, "kl_degree": 0.0}},
        "summary": {
            "id_linear_er20": {
                "e-shd_mean": 11.0,
                "e-shd_sem": 0.5,
                "e-shd_std": 1.0,
            }
        },
        "raw": raw_data,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(payload, f)


# ── ood_category ────────────────────────────────────────────────────────


class TestOODCategory:
    def test_id(self) -> None:
        assert ood_category("id_linear_er20") == "ID"

    def test_ood_graph(self) -> None:
        assert ood_category("ood_graph_sbm_linear") == "OOD-Graph"

    def test_ood_mech(self) -> None:
        assert ood_category("ood_mech_periodic_er40") == "OOD-Mech"

    def test_ood_both(self) -> None:
        assert ood_category("ood_both_sbm_periodic") == "OOD-Both"

    def test_ood_noise(self) -> None:
        assert ood_category("ood_noise_laplace_linear_er20") == "OOD-Noise"

    def test_ood_nodes(self) -> None:
        assert ood_category("ood_nodes_linear_er20_d40_n500") == "OOD-Nodes"

    def test_ood_samples(self) -> None:
        assert ood_category("ood_samples_linear_er20_d20_n1000") == "OOD-Samples"

    def test_fallback(self) -> None:
        assert ood_category("something_unknown") == "OOD"


# ── classify_failure_modes ──────────────────────────────────────────────


class TestClassifyFailureModes:
    def test_empty_input(self) -> None:
        result = classify_failure_modes(pd.DataFrame())
        assert result.empty

    def test_missing_metrics(self) -> None:
        """If required metrics are absent, return empty."""
        df = _make_raw_task_df([1.0], [0.9], [0.8])
        # Remove skeleton_f1
        df = df[df["Metric"] != "skeleton_f1"]
        result = classify_failure_modes(df)
        assert result.empty

    def test_empty_graph_classified(self) -> None:
        """sparsity_ratio ≈ 0 → empty."""
        df = _make_raw_task_df(
            sparsity_ratios=[0.01],
            skeleton_f1s=[0.0],
            orientation_accs=[0.0],
        )
        result = classify_failure_modes(df)
        assert len(result) == 1
        assert result["FailureMode"].iloc[0] == "empty"

    def test_dense_graph_classified(self) -> None:
        """sparsity_ratio > 2 → dense."""
        df = _make_raw_task_df(
            sparsity_ratios=[3.5],
            skeleton_f1s=[0.5],
            orientation_accs=[0.5],
        )
        result = classify_failure_modes(df)
        assert len(result) == 1
        assert result["FailureMode"].iloc[0] == "dense"

    def test_reversed_classified(self) -> None:
        """High skeleton_f1, low orientation_accuracy → reversed."""
        df = _make_raw_task_df(
            sparsity_ratios=[1.0],
            skeleton_f1s=[0.8],
            orientation_accs=[0.2],
        )
        result = classify_failure_modes(df)
        assert len(result) == 1
        assert result["FailureMode"].iloc[0] == "reversed"

    def test_sparse_classified(self) -> None:
        """sparsity_ratio in (0.05, 0.5) → sparse."""
        df = _make_raw_task_df(
            sparsity_ratios=[0.3],
            skeleton_f1s=[0.3],
            orientation_accs=[0.5],
        )
        result = classify_failure_modes(df)
        assert len(result) == 1
        assert result["FailureMode"].iloc[0] == "sparse"

    def test_reasonable_classified(self) -> None:
        """Normal prediction → reasonable."""
        df = _make_raw_task_df(
            sparsity_ratios=[1.0],
            skeleton_f1s=[0.9],
            orientation_accs=[0.8],
        )
        result = classify_failure_modes(df)
        assert len(result) == 1
        assert result["FailureMode"].iloc[0] == "reasonable"

    def test_multiple_tasks(self) -> None:
        """Multiple tasks get independently classified."""
        df = _make_raw_task_df(
            sparsity_ratios=[0.01, 3.5, 1.0],  # empty, dense, reasonable
            skeleton_f1s=[0.0, 0.5, 0.9],
            orientation_accs=[0.0, 0.5, 0.8],
        )
        result = classify_failure_modes(df)
        assert len(result) == 3
        modes = result.sort_values("TaskIdx")["FailureMode"].tolist()
        assert modes == ["empty", "dense", "reasonable"]

    def test_priority_empty_over_sparse(self) -> None:
        """Empty takes priority over sparse (both sparsity_ratio < 0.5)."""
        df = _make_raw_task_df(
            sparsity_ratios=[0.02],
            skeleton_f1s=[0.3],
            orientation_accs=[0.5],
        )
        result = classify_failure_modes(df)
        assert result["FailureMode"].iloc[0] == "empty"


# ── failure_mode_fractions ──────────────────────────────────────────────


class TestFailureModeFractions:
    def test_empty_input(self) -> None:
        result = failure_mode_fractions(pd.DataFrame())
        assert result.empty

    def test_fractions_sum_to_one(self) -> None:
        df = _make_raw_task_df(
            sparsity_ratios=[0.01, 3.5, 1.0, 1.0, 0.3],
            skeleton_f1s=[0.0, 0.5, 0.9, 0.8, 0.3],
            orientation_accs=[0.0, 0.5, 0.8, 0.2, 0.5],
        )
        classified = classify_failure_modes(df)
        fractions = failure_mode_fractions(classified, group_cols=("Model",))
        assert not fractions.empty
        row_sums = fractions[FAILURE_MODE_CATEGORIES].sum(axis=1)
        for s in row_sums:
            assert abs(s - 1.0) < 1e-9

    def test_all_categories_present(self) -> None:
        """Even if some categories have zero count, columns exist."""
        df = _make_raw_task_df(
            sparsity_ratios=[1.0],
            skeleton_f1s=[0.9],
            orientation_accs=[0.8],
        )
        classified = classify_failure_modes(df)
        fractions = failure_mode_fractions(classified, group_cols=("Model",))
        for cat in FAILURE_MODE_CATEGORIES:
            assert cat in fractions.columns


# ── load_raw_task_dataframe ─────────────────────────────────────────────


class TestLoadRawTaskDataframe:
    def test_basic_loading(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        _make_metrics_json(run_dir)
        df = load_raw_task_dataframe([run_dir])
        assert not df.empty
        assert "TaskIdx" in df.columns
        assert "Metric" in df.columns
        assert "Value" in df.columns

    def test_filters_metrics(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        _make_metrics_json(run_dir)
        df = load_raw_task_dataframe([run_dir], metrics=["sparsity_ratio"])
        assert set(df["Metric"].unique()) == {"sparsity_ratio"}

    def test_task_count(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        _make_metrics_json(run_dir)
        df = load_raw_task_dataframe([run_dir], metrics=["e-shd"])
        # 3 tasks in the synthetic data
        assert len(df) == 3
        assert sorted(df["TaskIdx"].unique()) == [0, 1, 2]

    def test_enrichment_columns(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run1"
        _make_metrics_json(run_dir)
        df = load_raw_task_dataframe([run_dir])
        assert "GraphType" in df.columns
        assert "SpectralDist" in df.columns
        assert df["GraphType"].iloc[0] == "er"

    def test_skips_prefixed_metrics(self, tmp_path: Path) -> None:
        """Metrics with '/' in the name are skipped."""
        run_dir = tmp_path / "run1"
        _make_metrics_json(
            run_dir,
            raw_data={
                "id_linear_er20": {
                    "e-shd": [10.0],
                    "id_linear_er20/e-shd": [10.0],
                }
            },
        )
        df = load_raw_task_dataframe([run_dir])
        assert "id_linear_er20/e-shd" not in df["Metric"].unique()
        assert "e-shd" in df["Metric"].unique()

    def test_empty_run_dirs(self) -> None:
        df = load_raw_task_dataframe([])
        assert df.empty


def test_pivot_raw_wide_preserves_rows_with_missing_enrichment() -> None:
    """Pivot keeps tasks even when optional enrichment columns are NaN."""
    rows: list[dict[str, object]] = []
    for task_idx in range(2):
        for metric, value in (
            ("fp_count", 1.0 + task_idx),
            ("fn_count", 2.0 + task_idx),
            ("reversed_count", 0.5 + task_idx),
        ):
            rows.append(
                {
                    "RunID": "run1",
                    "Model": "AviCi",
                    "ModelKey": "avici",
                    "DatasetKey": "ood_graph_sbm_linear_d20_n500",
                    "Dataset": "OOD-G SBM Linear",
                    "TaskIdx": task_idx,
                    "Metric": metric,
                    "Value": value,
                    "GraphType": "sbm",
                    "MechType": "linear",
                    "NNodes": 20,
                    "SamplesPerTask": 500,
                    "SparsityParam": None,
                    "SpectralDist": 1.2,
                    "KLDegreeDist": 2.1,
                }
            )

    raw_df = pd.DataFrame(rows)
    wide = _pivot_raw_wide(raw_df)

    assert len(wide) == 2
    assert set(wide["TaskIdx"].tolist()) == {0, 1}
    assert {"fp_count", "fn_count", "reversed_count"}.issubset(set(wide.columns))


# ── generate_failure_mode_bar (smoke test) ──────────────────────────────


def test_failure_mode_bar_creates_file(tmp_path: Path) -> None:
    from causal_meta.analysis.plots.results import generate_failure_mode_bar

    df = _make_raw_task_df(
        sparsity_ratios=[0.01, 3.5, 1.0, 0.3, 1.0],
        skeleton_f1s=[0.0, 0.5, 0.9, 0.3, 0.8],
        orientation_accs=[0.0, 0.5, 0.8, 0.5, 0.2],
    )
    classified = classify_failure_modes(df)
    fractions = failure_mode_fractions(classified, group_cols=("Model", "DatasetKey"))
    out = tmp_path / "fm.png"
    generate_failure_mode_bar(fractions, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_failure_mode_bar_handles_empty(tmp_path: Path) -> None:
    from causal_meta.analysis.plots.results import generate_failure_mode_bar

    out = tmp_path / "fm.png"
    generate_failure_mode_bar(pd.DataFrame(), out)
    assert not out.exists()
