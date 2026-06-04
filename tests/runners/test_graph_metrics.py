import numpy as np
import torch

from causal_meta.runners.metrics.graph import (
    ancestor_f1_score, auc_graph_scores_configurable, edge_entropy,
    expected_f1_score, expected_shd, graph_nll_per_edge_score,
    normalized_expected_shd, structural_interventional_distance,
    threshold_edge_probabilities, threshold_valid_dag_rate_from_probs)


def test_ancestor_f1() -> None:
    # 0 -> 1 -> 2
    target = torch.tensor([[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]).float()

    # Correct structure
    pred_correct = torch.tensor([[[[0, 1, 0], [0, 0, 1], [0, 0, 0]]]]).float()
    # Missing 1->2 but adds 0->2 directly
    pred_shortcut = torch.tensor([[[[0, 0, 1], [0, 0, 0], [0, 0, 0]]]]).float()

    # For target, ancestors are (0,1), (1,2), (0,2)
    # Correct pred: ancestors match.
    f1_correct = ancestor_f1_score(target, pred_correct)
    assert f1_correct[0].item() == 1.0

    # Shortcut pred: ancestors are (0,2).
    # TP = {(0,2)}, FP = {}, FN = {(0,1), (1,2)}
    # F1 = 2*1 / (2*1 + 0 + 2) = 2/4 = 0.5
    f1_shortcut = ancestor_f1_score(target, pred_shortcut)
    assert np.isclose(f1_shortcut[0].item(), 0.5)


def test_expected_shd_and_expected_f1() -> None:
    target = torch.tensor([[[0, 1], [0, 0]]])  # (B=1, N=2, N=2)
    pred = torch.tensor(
        [
            [[[0, 1], [0, 0]]],  # perfect
            [[[0, 0], [1, 0]]],  # reversed
        ]
    )  # (S=2, B=1, N=2, N=2)

    shd = expected_shd(target, pred)
    assert shd.shape == (1,)
    # SHD: perfect=0, reversed=1 (if not double) or 2 (if double).
    # Current expected_shd uses abs diff which is double counting mismatches in DAG context if considering reversed edge as 2 errors (FP+FN).
    # target=[0,1], pred=[0,0] -> FN=1
    # target=[0,0], pred=[1,0] -> FP=1
    # Total diff = 2.
    # So for sample 2, SHD=2.
    # Mean SHD = (0 + 2) / 2 = 1.0.
    assert shd[0].item() == 1.0

    f1 = expected_f1_score(target, pred)
    assert f1.shape == (1,)
    # Sample 1: Perfect match -> F1=1.0
    # Sample 2:
    # TP = 0 (target has (0,1), pred has (1,0))
    # FP = 1 (pred has (1,0))
    # FN = 1 (target has (0,1))
    # F1 = 0 / (0 + 1 + 1) = 0.0
    # Mean F1 = (1.0 + 0.0) / 2 = 0.5
    assert np.isclose(f1[0].item(), 0.5)


def test_auc_graph_scores_default_is_perfect_for_perfect_predictions() -> None:
    targets = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    preds = torch.stack([targets, targets, targets], dim=0)

    auc = auc_graph_scores_configurable(targets, preds)
    assert auc.shape == (1,)
    assert np.isclose(auc[0].item(), 1.0)


def test_auc_graph_scores_balanced_shuffling_is_deterministic() -> None:
    targets = torch.tensor([[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    probs = torch.tensor(
        [
            [[0.0, 0.65, 0.3], [0.2, 0.0, 0.2], [0.2, 0.2, 0.0]],
            [[0.0, 0.55, 0.35], [0.3, 0.0, 0.3], [0.3, 0.3, 0.0]],
        ]
    )
    preds = probs.unsqueeze(1)

    auc_one = auc_graph_scores_configurable(
        targets,
        preds,
        num_shuffles=64,
        balance_classes=True,
        seed=123,
    )
    auc_two = auc_graph_scores_configurable(
        targets,
        preds,
        num_shuffles=64,
        balance_classes=True,
        seed=123,
    )

    assert torch.allclose(auc_one, auc_two)
    assert 0.0 <= auc_one[0].item() <= 1.0


def test_normalized_expected_shd_is_size_comparable_for_full_mismatch() -> None:
    target_small = torch.zeros(1, 2, 2)
    pred_small = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    target_large = torch.zeros(1, 4, 4)
    pred_large = torch.ones(1, 1, 4, 4) - torch.eye(4).view(1, 1, 4, 4)

    shd_small = expected_shd(target_small, pred_small)
    shd_large = expected_shd(target_large, pred_large)
    ne_small = normalized_expected_shd(target_small, pred_small)
    ne_large = normalized_expected_shd(target_large, pred_large)

    assert shd_small[0].item() == 2.0
    assert shd_large[0].item() == 12.0
    assert np.isclose(ne_small[0].item(), 1.0)
    assert np.isclose(ne_large[0].item(), 1.0)


def test_graph_nll_per_edge_is_size_comparable_and_masks_diagonal() -> None:
    target_small = torch.zeros(1, 2, 2)
    target_large = torch.zeros(1, 4, 4)

    probs_small = torch.tensor([[[0.99, 0.25], [0.25, 0.99]]])
    probs_large = torch.full((1, 4, 4), 0.25)
    probs_large[:, torch.arange(4), torch.arange(4)] = 0.99

    preds_small = probs_small.unsqueeze(0)
    preds_large = probs_large.unsqueeze(0)

    nll_small = graph_nll_per_edge_score(target_small, preds_small)
    nll_large = graph_nll_per_edge_score(target_large, preds_large)
    expected = -np.log(0.75)

    assert np.isclose(nll_small, expected, atol=1e-6)
    assert np.isclose(nll_large, expected, atol=1e-6)


def test_edge_entropy_is_normalized_by_log_two() -> None:
    target_probs = torch.tensor([[[0.0, 0.5], [0.5, 0.0]]], dtype=torch.float32)
    preds = target_probs.unsqueeze(0)

    entropy = edge_entropy(preds, edge_probs=target_probs)

    assert np.isclose(entropy.item(), 1.0, atol=1e-6)


def test_threshold_edge_probabilities_returns_single_hard_graph() -> None:
    probs = torch.tensor([[[0.0, 0.6], [0.5, 0.0]]], dtype=torch.float32)
    thresholded = threshold_edge_probabilities(probs)

    assert thresholded.shape == (1, 1, 2, 2)
    assert thresholded[0, 0, 0, 1].item() == 1.0
    assert thresholded[0, 0, 1, 0].item() == 0.0


def test_threshold_valid_dag_rate_from_probs_uses_direct_probabilities() -> None:
    target = torch.zeros((1, 3, 3), dtype=torch.float32)
    probs = torch.tensor(
        [[[0.0, 0.9, 0.2], [0.1, 0.0, 0.8], [0.2, 0.1, 0.0]]],
        dtype=torch.float32,
    )

    rate = threshold_valid_dag_rate_from_probs(target, probs)
    assert rate.shape == (1,)
    assert rate[0].item() == 1.0


def test_sid_excludes_cyclic_predicted_samples() -> None:
    target = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
    valid_wrong = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
    cyclic = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    preds = torch.stack([valid_wrong, cyclic], dim=0)

    sid = structural_interventional_distance(target, preds)
    valid_only_sid = structural_interventional_distance(
        target, valid_wrong.unsqueeze(0)
    )

    assert torch.allclose(sid, valid_only_sid)
