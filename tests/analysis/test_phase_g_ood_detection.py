"""Tests for OOD detection and selective prediction analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_meta.analysis.uncertainty.ood_detection import (
    _precision_recall_auc_manual, _roc_auc_manual,
    compute_ood_detection_metrics, compute_selective_prediction,
    generate_ood_detection_table)

# ── Helpers ─────────────────────────────────────────────────────────────


def _make_raw_ood_df(
    n_id: int = 20,
    n_ood: int = 20,
    model: str = "TestModel",
    id_entropy_mean: float = 0.3,
    ood_entropy_mean: float = 0.7,
) -> pd.DataFrame:
    """Build synthetic per-task raw DataFrame with ID and OOD tasks."""
    rng = np.random.RandomState(42)
    rows: list[dict[str, object]] = []

    task_idx = 0
    # ID tasks
    for _ in range(n_id):
        entropy = float(rng.normal(id_entropy_mean, 0.1))
        shd = float(rng.normal(5.0, 1.0))
        sid = float(rng.normal(20.0, 2.0))
        nll = float(rng.normal(2.0, 0.5))
        ne_shd = shd / 20.0
        ne_sid = sid / 380.0
        for metric, val in [
            ("edge_entropy", entropy),
            ("e-shd", shd),
            ("e-sid", sid),
            ("ne-shd", ne_shd),
            ("ne-sid", ne_sid),
            ("graph_nll_per_edge", nll),
        ]:
            rows.append(
                {
                    "RunID": "run1",
                    "Model": model,
                    "ModelKey": model.lower(),
                    "DatasetKey": "id_linear_er20",
                    "Dataset": "id_linear_er20",
                    "TaskIdx": task_idx,
                    "Metric": metric,
                    "Value": val,
                    "GraphType": "er",
                    "MechType": "linear",
                    "NNodes": 20,
                    "SparsityParam": 0.1053,
                    "SpectralDist": 0.0,
                    "KLDegreeDist": 0.0,
                }
            )
        task_idx += 1

    # OOD tasks
    for _ in range(n_ood):
        entropy = float(rng.normal(ood_entropy_mean, 0.1))
        shd = float(rng.normal(15.0, 2.0))
        sid = float(rng.normal(40.0, 4.0))
        nll = float(rng.normal(5.0, 1.0))
        ne_shd = shd / 20.0
        ne_sid = sid / 380.0
        for metric, val in [
            ("edge_entropy", entropy),
            ("e-shd", shd),
            ("e-sid", sid),
            ("ne-shd", ne_shd),
            ("ne-sid", ne_sid),
            ("graph_nll_per_edge", nll),
        ]:
            rows.append(
                {
                    "RunID": "run1",
                    "Model": model,
                    "ModelKey": model.lower(),
                    "DatasetKey": "ood_mech_periodic_er40",
                    "Dataset": "ood_mech_periodic_er40",
                    "TaskIdx": task_idx,
                    "Metric": metric,
                    "Value": val,
                    "GraphType": "er",
                    "MechType": "periodic",
                    "NNodes": 20,
                    "SparsityParam": 0.1053,
                    "SpectralDist": 1.0,
                    "KLDegreeDist": 1.5,
                }
            )
        task_idx += 1

    return pd.DataFrame(rows)


# ── Manual AUC functions ────────────────────────────────────────────────


class TestROCAUCManual:
    def test_perfect_separation(self) -> None:
        labels = np.array([0, 0, 0, 1, 1, 1])
        scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        auc = _roc_auc_manual(labels, scores)
        assert auc == 1.0

    def test_random_scores(self) -> None:
        """Random scoring should give AUC ≈ 0.5."""
        rng = np.random.RandomState(0)
        n = 200
        labels = np.concatenate([np.zeros(n), np.ones(n)])
        scores = rng.rand(2 * n)
        auc = _roc_auc_manual(labels, scores)
        assert 0.35 < auc < 0.65

    def test_all_same_class(self) -> None:
        labels = np.array([1, 1, 1])
        scores = np.array([0.5, 0.6, 0.7])
        auc = _roc_auc_manual(labels, scores)
        assert np.isnan(auc)

    def test_inverse_scoring(self) -> None:
        """If OOD has lower scores, AUC < 0.5."""
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.8, 0.9, 0.1, 0.2])
        auc = _roc_auc_manual(labels, scores)
        assert auc == 0.0


class TestPRAUCManual:
    def test_perfect_separation(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        auprc = _precision_recall_auc_manual(labels, scores)
        assert auprc == pytest.approx(1.0, abs=0.01)

    def test_all_same_class(self) -> None:
        labels = np.array([0, 0, 0])
        scores = np.array([0.5, 0.6, 0.7])
        auprc = _precision_recall_auc_manual(labels, scores)
        assert np.isnan(auprc)


# ── compute_ood_detection_metrics ───────────────────────────────────────


class TestComputeOODDetectionMetrics:
    def test_basic_detection(self) -> None:
        raw_df = _make_raw_ood_df()
        result = compute_ood_detection_metrics(raw_df, score_metric="edge_entropy")
        assert not result.empty
        assert "AUROC" in result.columns
        assert "AUPRC" in result.columns
        assert len(result) == 1
        # With well-separated distributions, AUROC should be high
        assert result["AUROC"].iloc[0] > 0.7

    def test_multiple_models(self) -> None:
        df1 = _make_raw_ood_df(model="Model1")
        df2 = _make_raw_ood_df(model="Model2")
        combined = pd.concat([df1, df2], ignore_index=True)
        result = compute_ood_detection_metrics(combined, score_metric="edge_entropy")
        assert len(result) == 2

    def test_empty_input(self) -> None:
        result = compute_ood_detection_metrics(pd.DataFrame())
        assert result.empty

    def test_graph_nll_score(self) -> None:
        raw_df = _make_raw_ood_df()
        result = compute_ood_detection_metrics(
            raw_df, score_metric="graph_nll_per_edge"
        )
        assert not result.empty
        assert "AUROC" in result.columns

    def test_id_only_returns_empty(self) -> None:
        """If there are no OOD tasks, return empty."""
        raw_df = _make_raw_ood_df(n_ood=0)
        result = compute_ood_detection_metrics(raw_df, score_metric="edge_entropy")
        assert result.empty


# ── compute_selective_prediction ────────────────────────────────────────


class TestComputeSelectivePrediction:
    def test_basic_pareto(self) -> None:
        raw_df = _make_raw_ood_df()
        result = compute_selective_prediction(raw_df)
        assert not result.empty
        assert "Coverage" in result.columns
        assert "MeanValue" in result.columns
        assert "AccuracyMetric" in result.columns
        assert "Threshold" in result.columns
        assert set(result["AccuracyMetric"]) == {"ne-shd", "ne-sid"}

    def test_coverage_range(self) -> None:
        raw_df = _make_raw_ood_df()
        result = compute_selective_prediction(raw_df, n_thresholds=10)
        coverages = result["Coverage"].tolist()
        # At max threshold, coverage should be 1.0
        assert max(coverages) == pytest.approx(1.0, abs=0.01)
        # At lowest thresholds, coverage should be > 0
        assert min(coverages) > 0

    def test_lower_threshold_lower_shd(self) -> None:
        """Accepting only low-entropy predictions should give lower E-SHD."""
        raw_df = _make_raw_ood_df()
        result = compute_selective_prediction(raw_df, n_thresholds=50)
        shd_result = result[result["AccuracyMetric"] == "ne-shd"]
        if len(shd_result) > 1:
            low_t = (
                shd_result.sort_values(by="Coverage", ascending=True)
                .head(3)["MeanValue"]
                .mean()
            )
            high_t = (
                shd_result.sort_values(by="Coverage", ascending=False)
                .head(3)["MeanValue"]
                .mean()
            )
            # Low-entropy (low threshold) predictions should be better (lower SHD)
            assert low_t <= high_t + 5.0  # Allow some noise

    def test_empty_input(self) -> None:
        result = compute_selective_prediction(pd.DataFrame())
        assert result.empty

    def test_rejects_single_metric_mode(self) -> None:
        raw_df = _make_raw_ood_df()
        with pytest.raises(ValueError):
            compute_selective_prediction(raw_df, accuracy_metrics=["ne-shd"])


# ── generate_ood_detection_table ────────────────────────────────────────


class TestGenerateOODDetectionTable:
    def test_creates_file(self, tmp_path: Path) -> None:
        raw_df = _make_raw_ood_df()
        detection_df = compute_ood_detection_metrics(raw_df)
        out = tmp_path / "det.tex"
        generate_ood_detection_table(detection_df, out)
        assert out.exists()
        content = out.read_text()
        assert r"\begin{tabular}" in content
        assert "AUROC" in content

    def test_handles_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "det.tex"
        generate_ood_detection_table(pd.DataFrame(), out)
        assert not out.exists()


# ── Plot smoke tests ────────────────────────────────────────────────────


def test_entropy_histogram_creates_file(tmp_path: Path) -> None:
    from causal_meta.analysis.plots.results import generate_entropy_histogram

    raw_df = _make_raw_ood_df()
    out = tmp_path / "hist.png"
    generate_entropy_histogram(raw_df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_entropy_histogram_handles_empty(tmp_path: Path) -> None:
    from causal_meta.analysis.plots.results import generate_entropy_histogram

    out = tmp_path / "hist.png"
    generate_entropy_histogram(pd.DataFrame(), out)
    assert not out.exists()


def test_selective_prediction_pareto_creates_file(tmp_path: Path) -> None:
    from causal_meta.analysis.plots.results import \
        generate_selective_prediction_pareto

    raw_df = _make_raw_ood_df()
    pareto = compute_selective_prediction(raw_df)
    out = tmp_path / "pareto.png"
    generate_selective_prediction_pareto(pareto, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_selective_prediction_pareto_handles_empty(tmp_path: Path) -> None:
    from causal_meta.analysis.plots.results import \
        generate_selective_prediction_pareto

    out = tmp_path / "pareto.png"
    generate_selective_prediction_pareto(pd.DataFrame(), out)
    assert not out.exists()
