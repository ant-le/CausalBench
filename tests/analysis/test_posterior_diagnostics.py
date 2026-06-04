"""Tests for posterior failure diagnostics (artifact loading + per-sample analysis)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from causal_meta.analysis.diagnostics.posterior import (
    DENSE_RATIO_THRESHOLD,
    EMPTY_DENSITY_THRESHOLD,
    ORIENTATION_WRONG_THRESHOLD,
    SKELETON_CORRECT_F1_THRESHOLD,
    _connected_components,
    _discover_artifacts,
    _graph_density,
    _skeleton,
    compute_event_probabilities,
    compute_per_sample_diagnostics,
    compute_posterior_summary,
    load_posterior_artifacts,
    run_posterior_diagnostics,
    run_posterior_diagnostics_from_runs,
)


# ── Helpers ─────────────────────────────────────────────────────────────


def _make_run_dir(
    tmp_path: Path,
    *,
    n_nodes: int = 5,
    n_samples: int = 10,
    n_tasks: int = 3,
    model_name: str = "test_model",
    dataset_key: str = "id_linear_er20",
    graph_gen: str = "identity",
    use_shared_cache: bool = False,
) -> Path:
    """Create a fake run directory with metrics.json and inference artifacts.

    Args:
        graph_gen: One of "identity" (pred == truth), "empty" (all zeros),
            "dense" (all ones), "random".
    """
    run_dir = tmp_path / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write metrics.json
    metadata = {
        "run_id": "run_001",
        "run_name": "test_run",
        "model_name": model_name,
        "output_dir": str(run_dir),
    }

    if use_shared_cache:
        cache_root = tmp_path / "shared_cache"
        metadata["inference_root"] = str(cache_root)
        metadata["inference_layout"] = "model_dataset"

    metrics = {
        "metadata": metadata,
        "family_metadata": {
            dataset_key: {
                "n_nodes": n_nodes,
                "graph_type": "er",
                "mech_type": "linear",
                "sparsity_param": 0.1053,
            }
        },
        "distances": {dataset_key: {"spectral": 0.0, "kl_degree": 0.0}},
        "summary": {
            dataset_key: {"e-shd_mean": 1.0, "e-shd_sem": 0.1, "e-shd_std": 0.3}
        },
        "raw": {dataset_key: {"e-shd": [1.0] * n_tasks}},
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f)

    # Create inference artifacts
    if use_shared_cache:
        inference_dir = tmp_path / "shared_cache" / model_name / dataset_key
    else:
        inference_dir = run_dir / "inference" / dataset_key
    inference_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    for task_idx in range(n_tasks):
        # Generate a random ground truth DAG (lower triangular → DAG)
        mask = torch.tril(torch.ones(n_nodes, n_nodes), diagonal=-1)
        true_adj = (torch.rand(n_nodes, n_nodes) > 0.6).float() * mask
        true_adj = true_adj.to(torch.uint8)

        # Generate posterior samples based on graph_gen mode
        if graph_gen == "identity":
            # Perfect predictions — all samples match truth
            graph_samples = (
                true_adj.unsqueeze(0).unsqueeze(0).expand(1, n_samples, -1, -1)
            )
        elif graph_gen == "empty":
            graph_samples = torch.zeros(
                1, n_samples, n_nodes, n_nodes, dtype=torch.uint8
            )
        elif graph_gen == "dense":
            # All edges present (except diagonal)
            full = torch.ones(n_nodes, n_nodes, dtype=torch.uint8)
            full.fill_diagonal_(0)
            graph_samples = full.unsqueeze(0).unsqueeze(0).expand(1, n_samples, -1, -1)
        else:  # random
            graph_samples = (torch.rand(1, n_samples, n_nodes, n_nodes) > 0.5).to(
                torch.uint8
            )

        artifact = {
            "seed": task_idx * 100,
            "idx": task_idx,
            "graph_samples": graph_samples.clone(),
            "true_adj": true_adj.clone(),
            "cache_dtype": "uint8",
            "cache_n_samples": n_samples,
        }
        torch.save(artifact, inference_dir / f"seed_{task_idx * 100}.pt")

    return run_dir


def _make_simple_adj(edges: list[tuple[int, int]], n: int) -> torch.Tensor:
    """Create an adjacency matrix from a list of directed edges."""
    adj = torch.zeros(n, n, dtype=torch.uint8)
    for i, j in edges:
        adj[i, j] = 1
    return adj


# ── Unit tests: primitive functions ────────────────────────────────────


class TestGraphDensity:
    def test_empty_graph(self) -> None:
        adj = torch.zeros(5, 5)
        assert _graph_density(adj).item() == pytest.approx(0.0)

    def test_full_graph(self) -> None:
        adj = torch.ones(5, 5)
        adj.fill_diagonal_(0)
        # 20 edges out of 5*4 = 20 potential → density = 1.0
        assert _graph_density(adj).item() == pytest.approx(1.0)

    def test_partial_graph(self) -> None:
        adj = torch.zeros(4, 4)
        adj[0, 1] = 1
        adj[1, 2] = 1
        # 2 edges out of 4*3 = 12 potential → density ≈ 0.1667
        assert _graph_density(adj).item() == pytest.approx(2 / 12)

    def test_batch(self) -> None:
        adj = torch.zeros(3, 4, 4)
        adj[0, 0, 1] = 1
        adj[1, 0, 1] = 1
        adj[1, 1, 2] = 1
        result = _graph_density(adj)
        assert result.shape == (3,)
        assert result[0].item() == pytest.approx(1 / 12)
        assert result[1].item() == pytest.approx(2 / 12)
        assert result[2].item() == pytest.approx(0.0)

    def test_single_node(self) -> None:
        adj = torch.zeros(1, 1)
        assert _graph_density(adj).item() == pytest.approx(0.0)

    def test_ignores_diagonal_entries(self) -> None:
        adj = torch.zeros(4, 4)
        adj.fill_diagonal_(1)
        assert _graph_density(adj).item() == pytest.approx(0.0)


class TestSkeleton:
    def test_directed_to_undirected(self) -> None:
        adj = torch.zeros(3, 3)
        adj[0, 1] = 1  # 0 → 1
        adj[2, 1] = 1  # 2 → 1
        skel = _skeleton(adj)
        assert skel[0, 1].item() == 1.0
        assert skel[1, 0].item() == 1.0  # symmetric
        assert skel[1, 2].item() == 1.0
        assert skel[2, 1].item() == 1.0
        assert skel[0, 2].item() == 0.0

    def test_already_symmetric(self) -> None:
        adj = torch.zeros(3, 3)
        adj[0, 1] = 1
        adj[1, 0] = 1
        skel = _skeleton(adj)
        assert skel[0, 1].item() == 1.0
        assert skel[1, 0].item() == 1.0

    def test_empty(self) -> None:
        adj = torch.zeros(4, 4)
        skel = _skeleton(adj)
        assert skel.sum().item() == 0.0


class TestConnectedComponents:
    def test_single_component(self) -> None:
        # Chain: 0 → 1 → 2
        adj = _make_simple_adj([(0, 1), (1, 2)], 3)
        assert _connected_components(adj) == 1

    def test_two_components(self) -> None:
        # 0 → 1, 2 isolated
        adj = _make_simple_adj([(0, 1)], 3)
        assert _connected_components(adj) == 2

    def test_all_isolated(self) -> None:
        adj = torch.zeros(4, 4, dtype=torch.uint8)
        assert _connected_components(adj) == 4

    def test_fully_connected(self) -> None:
        adj = torch.ones(3, 3, dtype=torch.uint8)
        adj.fill_diagonal_(0)
        assert _connected_components(adj) == 1

    def test_empty_graph(self) -> None:
        adj = torch.zeros(0, 0, dtype=torch.uint8)
        assert _connected_components(adj) == 0


# ── Unit tests: per-sample diagnostics ─────────────────────────────────


class TestPerSampleDiagnostics:
    def test_perfect_predictions(self) -> None:
        """All samples match truth → density_ratio=1, skeleton_f1=1, orient_acc=1."""
        true_adj = _make_simple_adj([(0, 1), (1, 2), (2, 3)], 4)
        samples = true_adj.unsqueeze(0).expand(5, -1, -1).float()
        diag = compute_per_sample_diagnostics(samples, true_adj)

        assert len(diag["density"]) == 5
        np.testing.assert_allclose(diag["density_ratio"], 1.0, atol=1e-6)
        np.testing.assert_allclose(diag["skeleton_f1"], 1.0, atol=1e-6)
        np.testing.assert_allclose(diag["orientation_accuracy"], 1.0, atol=1e-6)

    def test_empty_predictions(self) -> None:
        """All-zero predictions → density=0, density_ratio≈0, skeleton_f1=0."""
        true_adj = _make_simple_adj([(0, 1), (1, 2)], 4)
        samples = torch.zeros(5, 4, 4)
        diag = compute_per_sample_diagnostics(samples, true_adj)

        np.testing.assert_allclose(diag["density"], 0.0, atol=1e-8)
        np.testing.assert_allclose(diag["density_ratio"], 0.0, atol=1e-6)
        # Empty skeleton vs non-empty truth → F1 should be low
        for f1 in diag["skeleton_f1"]:
            assert f1 < 0.5
        # All nodes isolated → cc = n_nodes
        np.testing.assert_array_equal(diag["connected_components"], 4)

    def test_reversed_predictions(self) -> None:
        """Predictions have correct skeleton but wrong direction."""
        true_adj = _make_simple_adj([(0, 1), (1, 2)], 3)
        reversed_adj = _make_simple_adj([(1, 0), (2, 1)], 3)  # reversed
        samples = reversed_adj.unsqueeze(0).expand(5, -1, -1).float()
        diag = compute_per_sample_diagnostics(samples, true_adj)

        # Skeleton should be perfect (same undirected edges)
        np.testing.assert_allclose(diag["skeleton_f1"], 1.0, atol=1e-6)
        # Orientation should be 0 (all edges reversed)
        np.testing.assert_allclose(diag["orientation_accuracy"], 0.0, atol=1e-6)

    def test_diagnostic_shapes(self) -> None:
        true_adj = _make_simple_adj([(0, 1)], 5)
        samples = torch.zeros(10, 5, 5)
        diag = compute_per_sample_diagnostics(samples, true_adj)

        assert diag["density"].shape == (10,)
        assert diag["density_ratio"].shape == (10,)
        assert diag["skeleton_f1"].shape == (10,)
        assert diag["orientation_accuracy"].shape == (10,)
        assert diag["connected_components"].shape == (10,)


# ── Unit tests: event probabilities ───────────────────────────────────


class TestEventProbabilities:
    def test_all_empty_samples(self) -> None:
        """All samples are empty → p_empty = 1.0."""
        true_adj = _make_simple_adj([(0, 1), (1, 2)], 4)
        samples = torch.zeros(20, 4, 4)
        diag = compute_per_sample_diagnostics(samples, true_adj)
        events = compute_event_probabilities(diag, true_adj)

        assert events["p_empty"] == pytest.approx(1.0)
        assert events["p_dense"] == pytest.approx(0.0)

    def test_all_dense_samples(self) -> None:
        """All samples are fully connected → p_dense should be high."""
        true_adj = _make_simple_adj([(0, 1)], 4)  # sparse truth
        full = torch.ones(4, 4, dtype=torch.uint8)
        full.fill_diagonal_(0)
        samples = full.unsqueeze(0).expand(20, -1, -1).float()
        diag = compute_per_sample_diagnostics(samples, true_adj)
        events = compute_event_probabilities(diag, true_adj)

        assert events["p_dense"] == pytest.approx(1.0)
        assert events["p_empty"] == pytest.approx(0.0)

    def test_perfect_predictions_no_failures(self) -> None:
        """Perfect predictions → all event probabilities near 0."""
        true_adj = _make_simple_adj([(0, 1), (1, 2), (2, 3)], 4)
        samples = true_adj.unsqueeze(0).expand(20, -1, -1).float()
        diag = compute_per_sample_diagnostics(samples, true_adj)
        events = compute_event_probabilities(diag, true_adj)

        assert events["p_empty"] == pytest.approx(0.0)
        assert events["p_dense"] == pytest.approx(0.0)
        assert events["p_skeleton_correct_orient_wrong"] == pytest.approx(0.0)
        assert events["p_fragmented"] == pytest.approx(0.0)

    def test_mixed_samples(self) -> None:
        """Mix of empty and perfect → event probabilities are fractions."""
        true_adj = _make_simple_adj([(0, 1), (1, 2)], 4)
        perfect = true_adj.unsqueeze(0).expand(5, -1, -1).float()
        empty = torch.zeros(5, 4, 4)
        samples = torch.cat([perfect, empty], dim=0)  # 10 samples: 5 perfect, 5 empty
        diag = compute_per_sample_diagnostics(samples, true_adj)
        events = compute_event_probabilities(diag, true_adj)

        assert events["p_empty"] == pytest.approx(0.5)

    def test_fragmented_predictions(self) -> None:
        """Predictions with more components than truth → p_fragmented > 0."""
        # Truth: chain 0→1→2→3 (1 component)
        true_adj = _make_simple_adj([(0, 1), (1, 2), (2, 3)], 4)
        # Prediction: only edge 0→1, nodes 2,3 isolated (3 components > 1)
        frag_adj = _make_simple_adj([(0, 1)], 4)
        samples = frag_adj.unsqueeze(0).expand(10, -1, -1).float()
        diag = compute_per_sample_diagnostics(samples, true_adj)
        events = compute_event_probabilities(diag, true_adj)

        assert events["p_fragmented"] == pytest.approx(1.0)

    def test_empty_diagnostics(self) -> None:
        diag = {
            "density": np.array([]),
            "density_ratio": np.array([]),
            "skeleton_f1": np.array([]),
            "orientation_accuracy": np.array([]),
            "connected_components": np.array([], dtype=np.int32),
        }
        true_adj = torch.zeros(3, 3, dtype=torch.uint8)
        events = compute_event_probabilities(diag, true_adj)
        assert events["p_empty"] == 0.0
        assert events["p_dense"] == 0.0

    def test_fragmented_is_nan_when_truth_disconnected(self) -> None:
        true_adj = _make_simple_adj([(0, 1), (2, 3)], 4)  # two components
        samples = true_adj.unsqueeze(0).expand(8, -1, -1).float()
        diag = compute_per_sample_diagnostics(samples, true_adj)
        events = compute_event_probabilities(diag, true_adj)

        assert bool(events["truth_connected"]) is False
        assert np.isnan(float(events["p_fragmented"]))


# ── Unit tests: posterior summary ─────────────────────────────────────


class TestPosteriorSummary:
    def test_basic_stats(self) -> None:
        diag = {
            "density": np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            "density_ratio": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        }
        summary = compute_posterior_summary(diag)

        assert "density" in summary
        assert summary["density"]["mean"] == pytest.approx(0.3)
        assert summary["density"]["median"] == pytest.approx(0.3)
        assert summary["density_ratio"]["mean"] == pytest.approx(1.0)
        assert summary["density_ratio"]["std"] == pytest.approx(0.0)

    def test_quantiles(self) -> None:
        diag = {"values": np.linspace(0, 100, 101)}
        summary = compute_posterior_summary(diag)

        assert summary["values"]["q25"] == pytest.approx(25.0)
        assert summary["values"]["q75"] == pytest.approx(75.0)
        assert summary["values"]["q05"] == pytest.approx(5.0)
        assert summary["values"]["q95"] == pytest.approx(95.0)

    def test_empty_array(self) -> None:
        diag = {"empty": np.array([])}
        summary = compute_posterior_summary(diag)
        assert np.isnan(summary["empty"]["mean"])
        assert np.isnan(summary["empty"]["std"])


# ── Integration tests: artifact loading ───────────────────────────────


class TestArtifactDiscovery:
    def test_discovers_artifacts(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, n_tasks=3)
        artifacts = _discover_artifacts(run_dir)
        assert len(artifacts) == 3
        assert all(dk == "id_linear_er20" for dk, _ in artifacts)

    def test_no_inference_dir(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "empty_run"
        run_dir.mkdir()
        assert _discover_artifacts(run_dir) == []

    def test_multiple_datasets(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "multi_run"
        run_dir.mkdir()
        for dk in ["dataset_a", "dataset_b"]:
            inf_dir = run_dir / "inference" / dk
            inf_dir.mkdir(parents=True)
            artifact = {
                "seed": 0,
                "idx": 0,
                "graph_samples": torch.zeros(1, 5, 3, 3),
                "true_adj": torch.zeros(3, 3),
            }
            torch.save(artifact, inf_dir / "seed_0.pt")

        artifacts = _discover_artifacts(run_dir)
        dataset_keys = [dk for dk, _ in artifacts]
        assert "dataset_a" in dataset_keys
        assert "dataset_b" in dataset_keys

    def test_discovers_shared_cache_layout(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, n_tasks=2, use_shared_cache=True)
        artifacts = _discover_artifacts(
            run_dir,
            model_name="test_model",
            inference_root=tmp_path / "shared_cache",
            use_model_subdir=True,
        )
        assert len(artifacts) == 2


class TestLoadPosteriorArtifacts:
    def test_loads_correctly(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, n_nodes=5, n_samples=8, n_tasks=2)
        df = load_posterior_artifacts([run_dir])

        assert len(df) == 2
        assert "GraphSamples" in df.columns
        assert "TrueAdj" in df.columns
        assert df.iloc[0]["NumSamples"] == 8
        assert df.iloc[0]["NNodes"] == 5
        assert df.iloc[0]["GraphSamples"].shape == (8, 5, 5)
        assert df.iloc[0]["TrueAdj"].shape == (5, 5)

    def test_filters_by_dataset_key(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, dataset_key="id_linear_er20")
        df = load_posterior_artifacts([run_dir], dataset_keys=["ood_graph_sbm_linear"])
        assert len(df) == 0

    def test_max_tasks_per_family(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, n_tasks=5)
        df = load_posterior_artifacts([run_dir], max_tasks_per_family=2)
        assert len(df) == 2

    def test_missing_metrics_json(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "no_metrics"
        run_dir.mkdir()
        df = load_posterior_artifacts([run_dir])
        assert df.empty

    def test_binary_enforcement(self, tmp_path: Path) -> None:
        """Loaded tensors should be binary uint8 regardless of stored dtype."""
        run_dir = _make_run_dir(tmp_path, n_nodes=4, n_samples=3, n_tasks=1)
        df = load_posterior_artifacts([run_dir])
        samples = df.iloc[0]["GraphSamples"]
        # Values should be 0 or 1
        unique_vals = torch.unique(samples)
        assert all(v.item() in (0, 1) for v in unique_vals)

    def test_loads_from_shared_cache_layout(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, n_tasks=2, use_shared_cache=True)
        df = load_posterior_artifacts([run_dir])
        assert len(df) == 2
        assert set(df["DatasetKey"]) == {"id_linear_er20"}


# ── Integration tests: full pipeline ──────────────────────────────────


class TestRunPosteriorDiagnostics:
    def test_identity_predictions(self, tmp_path: Path) -> None:
        """Perfect predictions should yield benign diagnostics."""
        run_dir = _make_run_dir(
            tmp_path, n_nodes=5, n_samples=10, n_tasks=2, graph_gen="identity"
        )
        artifacts_df = load_posterior_artifacts([run_dir])
        result = run_posterior_diagnostics(artifacts_df)

        assert len(result) == 2
        assert "p_empty" in result.columns
        assert "p_dense" in result.columns
        assert "density_ratio_mean" in result.columns
        assert "skeleton_f1_mean" in result.columns

        # Perfect predictions: events should all be 0
        for _, row in result.iterrows():
            assert row["p_empty"] == pytest.approx(0.0)
            assert row["p_dense"] == pytest.approx(0.0)
            assert row["p_skeleton_correct_orient_wrong"] == pytest.approx(0.0)

    def test_empty_predictions(self, tmp_path: Path) -> None:
        """All-zero predictions should trigger p_empty = 1."""
        run_dir = _make_run_dir(
            tmp_path, n_nodes=5, n_samples=10, n_tasks=2, graph_gen="empty"
        )
        artifacts_df = load_posterior_artifacts([run_dir])
        result = run_posterior_diagnostics(artifacts_df)

        for _, row in result.iterrows():
            assert row["p_empty"] == pytest.approx(1.0)

    def test_dense_predictions(self, tmp_path: Path) -> None:
        """All-ones predictions should trigger p_dense > 0."""
        run_dir = _make_run_dir(
            tmp_path, n_nodes=5, n_samples=10, n_tasks=2, graph_gen="dense"
        )
        artifacts_df = load_posterior_artifacts([run_dir])
        result = run_posterior_diagnostics(artifacts_df)

        for _, row in result.iterrows():
            assert row["p_dense"] == pytest.approx(1.0)

    def test_empty_input(self) -> None:
        result = run_posterior_diagnostics(pd.DataFrame())
        assert result.empty

    def test_result_columns(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, n_tasks=1)
        artifacts_df = load_posterior_artifacts([run_dir])
        result = run_posterior_diagnostics(artifacts_df)

        expected_cols = {
            "RunID",
            "RunDir",
            "Model",
            "DatasetKey",
            "Seed",
            "TaskIdx",
            "NumSamples",
            "NNodes",
            "p_empty",
            "p_dense",
            "p_skeleton_correct_orient_wrong",
            "p_fragmented",
            "density_mean",
            "density_std",
            "density_median",
            "density_ratio_mean",
            "density_ratio_std",
            "skeleton_f1_mean",
            "skeleton_f1_std",
            "orientation_accuracy_mean",
            "orientation_accuracy_std",
            "connected_components_mean",
            "connected_components_std",
        }
        assert expected_cols.issubset(set(result.columns))


class TestEndToEndPipeline:
    def test_from_runs(self, tmp_path: Path) -> None:
        """Test the convenience wrapper."""
        run_dir = _make_run_dir(
            tmp_path, n_nodes=4, n_samples=5, n_tasks=2, graph_gen="random"
        )
        result = run_posterior_diagnostics_from_runs([run_dir])

        assert len(result) == 2
        assert "p_empty" in result.columns
        # Random predictions — just check types and ranges
        for _, row in result.iterrows():
            assert 0.0 <= row["p_empty"] <= 1.0
            assert 0.0 <= row["p_dense"] <= 1.0
            if bool(row.get("TruthConnected", False)):
                assert 0.0 <= row["p_fragmented"] <= 1.0
            else:
                assert np.isnan(float(row["p_fragmented"]))
            assert row["density_ratio_mean"] >= 0.0

    def test_with_dataset_filter(self, tmp_path: Path) -> None:
        run_dir = _make_run_dir(tmp_path, n_tasks=2, dataset_key="id_linear_er20")
        result = run_posterior_diagnostics_from_runs(
            [run_dir], dataset_keys=["id_linear_er20"]
        )
        assert len(result) == 2

        result_empty = run_posterior_diagnostics_from_runs(
            [run_dir], dataset_keys=["nonexistent"]
        )
        assert result_empty.empty


# ── Plot smoke tests ──────────────────────────────────────────────────


class TestPosteriorDiagnosticPlots:
    def test_event_probability_bar(self, tmp_path: Path) -> None:
        """Smoke test: event probability bar chart renders without error."""
        from causal_meta.analysis.plots.results import generate_event_probability_bar

        run_dir = _make_run_dir(
            tmp_path / "run", n_nodes=4, n_samples=5, n_tasks=3, graph_gen="random"
        )
        diag_df = run_posterior_diagnostics_from_runs([run_dir])
        out = tmp_path / "event_prob.png"
        generate_event_probability_bar(diag_df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_event_probability_bar_empty(self, tmp_path: Path) -> None:
        from causal_meta.analysis.plots.results import generate_event_probability_bar

        out = tmp_path / "event_prob_empty.png"
        generate_event_probability_bar(pd.DataFrame(), out)
        assert not out.exists()

    def test_posterior_diagnostic_violins(self, tmp_path: Path) -> None:
        """Smoke test: violin plot renders without error."""
        from causal_meta.analysis.plots.results import (
            generate_posterior_diagnostic_violins,
        )

        run_dir = _make_run_dir(
            tmp_path / "run", n_nodes=4, n_samples=5, n_tasks=3, graph_gen="random"
        )
        diag_df = run_posterior_diagnostics_from_runs([run_dir])
        out = tmp_path / "violins.png"
        generate_posterior_diagnostic_violins(diag_df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_posterior_diagnostic_violins_empty(self, tmp_path: Path) -> None:
        from causal_meta.analysis.plots.results import (
            generate_posterior_diagnostic_violins,
        )

        out = tmp_path / "violins_empty.png"
        generate_posterior_diagnostic_violins(pd.DataFrame(), out)
        assert not out.exists()
