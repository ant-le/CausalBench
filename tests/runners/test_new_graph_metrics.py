"""Tests for Phase A metrics: edge confusion decomposition, sparsity ratio,
skeleton/orientation split, and expected calibration error."""

from __future__ import annotations

import torch

from causal_meta.runners.metrics.graph import (
    Metrics,
    edge_confusion_decomposition,
    expected_shd,
    expected_calibration_error,
    skeleton_orientation_scores,
    sparsity_ratio,
    valid_dag_rate,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _make_target_pred(
    target_list: list[list[list[int]]],
    pred_list: list[list[list[list[int]]]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build (B,N,N) target and (S,B,N,N) pred tensors from nested lists."""
    return (
        torch.tensor(target_list, dtype=torch.float32),
        torch.tensor(pred_list, dtype=torch.float32),
    )


# ── edge_confusion_decomposition ──────────────────────────────────────


class TestEdgeConfusionDecomposition:
    """Tests for the edge confusion decomposition metric."""

    def test_perfect_prediction(self) -> None:
        # Single graph: 0->1
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.float32)

        decomp = edge_confusion_decomposition(target, pred)
        assert decomp["correct"][0].item() == 1.0
        assert decomp["fp"][0].item() == 0.0
        assert decomp["fn"][0].item() == 0.0
        assert decomp["reversed"][0].item() == 0.0

    def test_reversed_edge(self) -> None:
        # Target: 0->1, Pred: 1->0 (reversed)
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 0], [1, 0]]]], dtype=torch.float32)

        decomp = edge_confusion_decomposition(target, pred)
        assert decomp["reversed"][0].item() == 1.0
        assert decomp["correct"][0].item() == 0.0
        assert decomp["fn"][0].item() == 0.0
        assert decomp["fp"][0].item() == 0.0

    def test_false_positive(self) -> None:
        # Target: empty graph, Pred: 0->1 (pure FP)
        target = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.float32)

        decomp = edge_confusion_decomposition(target, pred)
        assert decomp["fp"][0].item() == 1.0
        assert decomp["correct"][0].item() == 0.0
        assert decomp["fn"][0].item() == 0.0
        assert decomp["reversed"][0].item() == 0.0

    def test_false_negative(self) -> None:
        # Target: 0->1, Pred: empty graph
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 0], [0, 0]]]], dtype=torch.float32)

        decomp = edge_confusion_decomposition(target, pred)
        assert decomp["fn"][0].item() == 1.0
        assert decomp["correct"][0].item() == 0.0
        assert decomp["fp"][0].item() == 0.0
        assert decomp["reversed"][0].item() == 0.0

    def test_averaging_over_samples(self) -> None:
        # Target: 0->1; Sample 1: correct, Sample 2: empty (FN)
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor(
            [
                [[[0, 1], [0, 0]]],  # correct
                [[[0, 0], [0, 0]]],  # miss
            ],
            dtype=torch.float32,
        )

        decomp = edge_confusion_decomposition(target, pred)
        assert abs(decomp["correct"][0].item() - 0.5) < 1e-6
        assert abs(decomp["fn"][0].item() - 0.5) < 1e-6

    def test_3node_mixed(self) -> None:
        # Target: 0->1, 1->2; Pred: 0->1 (correct), 2->1 (reversed)
        target = torch.tensor([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1, 0], [0, 0, 0], [0, 1, 0]]]], dtype=torch.float32)

        decomp = edge_confusion_decomposition(target, pred)
        assert decomp["correct"][0].item() == 1.0  # 0->1
        assert decomp["reversed"][0].item() == 1.0  # 2->1 is reversed 1->2
        assert decomp["fn"][0].item() == 0.0
        assert decomp["fp"][0].item() == 0.0

    def test_bidirectional_overprediction_counts_extra_fp(self) -> None:
        # Target: 0->1, Pred: 0->1 and 1->0
        # Extra reverse edge should be counted as FP, not reversed.
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float32)

        decomp = edge_confusion_decomposition(target, pred)
        assert decomp["correct"][0].item() == 1.0
        assert decomp["fp"][0].item() == 1.0
        assert decomp["fn"][0].item() == 0.0
        assert decomp["reversed"][0].item() == 0.0

    def test_decomposition_matches_expected_shd_identity(self) -> None:
        # For this decomposition, SHD identity should hold exactly:
        #   SHD = FP + FN + 2 * Reversed
        target = torch.tensor(
            [
                [
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, 0, 0],
                ]
            ],
            dtype=torch.float32,
        )
        pred = torch.tensor(
            [
                [
                    [
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                    ]
                ],
                [
                    [
                        [0, 0, 0],
                        [1, 0, 1],
                        [0, 0, 0],
                    ]
                ],
            ],
            dtype=torch.float32,
        )

        decomp = edge_confusion_decomposition(target, pred)
        shd = expected_shd(target, pred)
        lhs = shd[0].item()
        rhs = (
            decomp["fp"][0].item()
            + decomp["fn"][0].item()
            + 2.0 * decomp["reversed"][0].item()
        )
        assert abs(lhs - rhs) < 1e-6

    def test_shape_validation(self) -> None:
        bad_target = torch.zeros(2, 3)
        bad_pred = torch.zeros(1, 2, 3, 3)
        try:
            edge_confusion_decomposition(bad_target, bad_pred)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ── sparsity_ratio ─────────────────────────────────────────────────────


class TestSparsityRatio:
    """Tests for the sparsity ratio metric."""

    def test_perfect_prediction_ratio_is_one(self) -> None:
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.float32)

        ratio = sparsity_ratio(target, pred)
        assert abs(ratio[0].item() - 1.0) < 1e-5

    def test_empty_prediction_ratio_near_zero(self) -> None:
        target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 0], [0, 0]]]], dtype=torch.float32)

        ratio = sparsity_ratio(target, pred)
        assert ratio[0].item() < 0.01

    def test_overprediction_ratio_above_one(self) -> None:
        # Target: 1 edge out of 2 possible (2-node), density = 0.5
        # Pred: 2 edges, density = 1.0 → ratio = 2.0
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.float32)

        ratio = sparsity_ratio(target, pred)
        assert abs(ratio[0].item() - 2.0) < 1e-5

    def test_empty_target_does_not_crash(self) -> None:
        target = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.float32)

        ratio = sparsity_ratio(target, pred)
        # Should not be NaN or inf
        assert torch.isfinite(ratio[0])


# ── skeleton_orientation_scores ────────────────────────────────────────


class TestSkeletonOrientationScores:
    """Tests for skeleton F1 and orientation accuracy."""

    def test_perfect_prediction(self) -> None:
        target = torch.tensor([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]], dtype=torch.float32)

        scores = skeleton_orientation_scores(target, pred)
        assert abs(scores["skeleton_f1"][0].item() - 1.0) < 1e-6
        assert abs(scores["orientation_accuracy"][0].item() - 1.0) < 1e-6

    def test_reversed_edge_skeleton_correct_orientation_wrong(self) -> None:
        # Target: 0->1, Pred: 1->0
        # Skeleton is correct (undirected edge between 0,1), orientation is wrong
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 0], [1, 0]]]], dtype=torch.float32)

        scores = skeleton_orientation_scores(target, pred)
        assert abs(scores["skeleton_f1"][0].item() - 1.0) < 1e-6
        assert abs(scores["orientation_accuracy"][0].item() - 0.0) < 1e-6

    def test_missing_edge_skeleton_wrong(self) -> None:
        # Target: 0->1, Pred: empty → skeleton FP=0, FN=1
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 0], [0, 0]]]], dtype=torch.float32)

        scores = skeleton_orientation_scores(target, pred)
        assert scores["skeleton_f1"][0].item() == 0.0
        # No common skeleton edges → orientation defaults to 1.0
        assert scores["orientation_accuracy"][0].item() == 1.0

    def test_extra_edge_skeleton_fp(self) -> None:
        # Target: 0->1, Pred: 0->1 + 0->2 (extra edge)
        target = torch.tensor([[[0, 1, 0], [0, 0, 0], [0, 0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1, 1], [0, 0, 0], [0, 0, 0]]]], dtype=torch.float32)

        scores = skeleton_orientation_scores(target, pred)
        # Skeleton: TP=1 (0-1), FP=1 (0-2), FN=0; F1 = 2*1/(2+1+0) = 2/3
        assert abs(scores["skeleton_f1"][0].item() - 2.0 / 3.0) < 1e-5
        # Common skeleton edges: only 0-1, and direction is correct
        assert abs(scores["orientation_accuracy"][0].item() - 1.0) < 1e-6

    def test_averaging_over_samples(self) -> None:
        # Target: 0->1; Sample 1: correct, Sample 2: reversed
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor(
            [
                [[[0, 1], [0, 0]]],  # correct direction
                [[[0, 0], [1, 0]]],  # reversed direction
            ],
            dtype=torch.float32,
        )

        scores = skeleton_orientation_scores(target, pred)
        # Both samples have correct skeleton → skeleton_f1 = 1.0
        assert abs(scores["skeleton_f1"][0].item() - 1.0) < 1e-6
        # Orientation: 1 correct + 0 wrong → average = 0.5
        assert abs(scores["orientation_accuracy"][0].item() - 0.5) < 1e-6


# ── expected_calibration_error ─────────────────────────────────────────


class TestExpectedCalibrationError:
    """Tests for ECE metric."""

    def test_perfectly_calibrated(self) -> None:
        # All probabilities match targets → ECE = 0
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        # All samples agree → probability = exactly 0 or 1
        pred = torch.tensor([[[[0, 1], [0, 0]]]] * 10, dtype=torch.float32)

        ece = expected_calibration_error(target, pred, n_bins=10)
        assert abs(ece.item()) < 1e-5

    def test_completely_wrong_and_confident(self) -> None:
        # Target: 0->1, Pred: always empty → prob(0->1)=0 but truth=1
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 0], [0, 0]]]] * 10, dtype=torch.float32)

        ece = expected_calibration_error(target, pred, n_bins=10)
        # ECE > 0 because model is confident (p=0) but wrong on one edge
        assert ece.item() > 0.0

    def test_ece_range(self) -> None:
        # ECE should always be in [0, 1]
        target = torch.randint(0, 2, (4, 5, 5), dtype=torch.float32)
        pred = torch.randint(0, 2, (8, 4, 5, 5), dtype=torch.float32)

        ece = expected_calibration_error(target, pred, n_bins=10)
        assert 0.0 <= ece.item() <= 1.0

    def test_shape_validation(self) -> None:
        bad_target = torch.zeros(2, 3)
        bad_pred = torch.zeros(1, 2, 3, 3)
        try:
            expected_calibration_error(bad_target, bad_pred)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ── valid_dag_rate ──────────────────────────────────────────────────────


class TestValidDagRate:
    """Tests for DAG-validity rate over posterior samples."""

    def test_all_samples_are_valid_dags(self) -> None:
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor(
            [
                [[[0, 1], [0, 0]]],
                [[[0, 0], [0, 0]]],
            ],
            dtype=torch.float32,
        )

        rate = valid_dag_rate(target, pred)
        assert abs(rate[0].item() - 1.0) < 1e-6

    def test_all_samples_are_invalid(self) -> None:
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor(
            [
                [[[0, 1], [1, 0]]],
                [[[1, 0], [0, 0]]],
            ],
            dtype=torch.float32,
        )

        rate = valid_dag_rate(target, pred)
        assert abs(rate[0].item() - 0.0) < 1e-6

    def test_mixed_validity_rate(self) -> None:
        target = torch.tensor(
            [
                [[0, 1], [0, 0]],
                [[0, 1], [0, 0]],
            ],
            dtype=torch.float32,
        )
        pred = torch.tensor(
            [
                [
                    [[0, 1], [0, 0]],
                    [[0, 1], [1, 0]],
                ],
                [
                    [[0, 1], [1, 0]],
                    [[0, 1], [0, 0]],
                ],
            ],
            dtype=torch.float32,
        )

        rate = valid_dag_rate(target, pred)
        assert abs(rate[0].item() - 0.5) < 1e-6
        assert abs(rate[1].item() - 0.5) < 1e-6


# ── Integration: Metrics class with new keys ───────────────────────────


class TestMetricsClassNewKeys:
    """Verify that the Metrics class dispatches the new metric keys."""

    def test_new_metrics_in_default_list(self) -> None:
        m = Metrics()
        # Count metrics are opt-in (excluded from defaults for cleaner validation).
        for key in [
            "sparsity_ratio",
            "skeleton_f1",
            "orientation_accuracy",
            "valid_dag_pct",
            "ece",
        ]:
            assert key in m.metrics_list, f"{key} not in default metrics_list"
        # Count metrics should NOT be in default list.
        for key in ["fp_count", "fn_count", "reversed_count", "correct_count"]:
            assert key not in m.metrics_list, (
                f"{key} should not be in default metrics_list"
            )

    def test_compute_batch_metrics_returns_new_keys(self) -> None:
        m = Metrics(
            metrics=[
                "fp_count",
                "fn_count",
                "reversed_count",
                "correct_count",
                "sparsity_ratio",
                "skeleton_f1",
                "orientation_accuracy",
                "valid_dag_pct",
                "sampled-valid_dag_pct",
                "ece",
            ]
        )
        target = torch.tensor([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]], dtype=torch.float32)

        result = m._compute_batch_metrics(target, pred)
        for key in [
            "fp_count",
            "fn_count",
            "reversed_count",
            "correct_count",
            "sparsity_ratio",
            "skeleton_f1",
            "orientation_accuracy",
            "valid_dag_pct",
            "sampled-valid_dag_pct",
            "ece",
        ]:
            assert key in result, f"{key} missing from batch metrics output"
            assert isinstance(result[key], float), f"{key} is not a float"
        assert abs(result["valid_dag_pct"] - 100.0) < 1e-6

    def test_threshold_valid_dag_pct_can_exceed_sample_valid_dag_pct(self) -> None:
        """A thresholded mean graph can be a DAG even when most samples are cyclic."""
        m = Metrics(
            metrics=[
                "valid_dag_pct",
                "sampled-valid_dag_pct",
                "threshold_valid_dag_pct",
            ]
        )
        target = torch.zeros((1, 3, 3), dtype=torch.float32)
        pred = torch.tensor(
            [
                [[[0, 1, 0], [0, 0, 1], [1, 0, 0]]],
                [[[0, 1, 0], [0, 0, 1], [0, 0, 0]]],
                [[[0, 1, 0], [0, 0, 1], [0, 0, 0]]],
            ],
            dtype=torch.float32,
        )

        edge_probs = torch.tensor(
            [[[0.0, 0.9, 0.1], [0.1, 0.0, 0.9], [0.4, 0.1, 0.0]]],
            dtype=torch.float32,
        )
        thresholded_graph = (edge_probs > 0.5).float().unsqueeze(0)

        result = m._compute_batch_metrics(
            target,
            pred,
            thresholded_graph=thresholded_graph,
            edge_probs=edge_probs,
        )
        assert abs(result["sampled-valid_dag_pct"] - (200.0 / 3.0)) < 1e-5
        assert abs(result["valid_dag_pct"] - 100.0) < 1e-6
        assert abs(result["threshold_valid_dag_pct"] - 100.0) < 1e-6

    def test_update_and_compute_flow(self) -> None:
        """Full update → compute cycle with new metrics."""
        m = Metrics(
            metrics=[
                "fp_count",
                "skeleton_f1",
                "valid_dag_pct",
                "threshold_valid_dag_pct",
                "ece",
            ]
        )
        target = torch.tensor([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]], dtype=torch.float32)

        m.update(target, pred)
        m.update(target, pred)

        result = m.compute(summary_stats=True)
        assert "fp_count_mean" in result
        assert "skeleton_f1_mean" in result
        assert "valid_dag_pct_mean" in result
        assert "threshold_valid_dag_pct_mean" in result
        assert "ece_mean" in result

    def test_selective_metrics_no_crash(self) -> None:
        """Only request a subset of new metrics — others should not run."""
        m = Metrics(metrics=["sparsity_ratio"])
        target = torch.tensor([[[0, 1], [0, 0]]], dtype=torch.float32)
        pred = torch.tensor([[[[0, 1], [0, 0]]]], dtype=torch.float32)

        result = m._compute_batch_metrics(target, pred)
        assert "sparsity_ratio" in result
        assert "fp_count" not in result
        assert "skeleton_f1" not in result
        assert "ece" not in result
