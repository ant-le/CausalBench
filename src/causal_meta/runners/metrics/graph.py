from __future__ import annotations

import math
from typing import Any, Dict, List

import torch

from .base import BaseMetrics


def _auc_from_binary_labels(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """Compute trapezoidal AUC from binary labels and prediction scores."""
    positives = float(y_true.sum().item())
    negatives = float(y_true.numel() - y_true.sum().item())
    if positives <= 0.0 or negatives <= 0.0:
        return 0.5

    sorted_indices = torch.argsort(y_score, descending=True)
    sorted_scores = y_score[sorted_indices]
    sorted_true = y_true[sorted_indices]

    tps = torch.cumsum(sorted_true, dim=0)
    fps = torch.cumsum(1.0 - sorted_true, dim=0)

    if sorted_scores.numel() == 0:
        return 0.5

    score_diff = sorted_scores[1:] != sorted_scores[:-1]
    change_indices = torch.nonzero(score_diff, as_tuple=False).flatten() + 1
    group_ends = torch.cat(
        [change_indices - 1, torch.tensor([sorted_scores.numel() - 1])]
    )

    tps = tps[group_ends]
    fps = fps[group_ends]

    tpr = torch.cat([torch.zeros(1), tps / positives])
    fpr = torch.cat([torch.zeros(1), fps / negatives])

    return float(torch.trapezoid(tpr, fpr).item())


def auc_graph_scores_configurable(
    targets: torch.Tensor,
    preds: torch.Tensor,
    *,
    num_shuffles: int = 1000,
    balance_classes: bool = True,
    seed: int = 0,
) -> torch.Tensor:
    """Compute AUC with configurable class balancing options.

    Args:
        targets: Binary adjacency matrices ``(batch_size, n_nodes, n_nodes)``.
        preds: Graph samples/probabilities
            ``(num_samples, batch_size, n_nodes, n_nodes)``.
        num_shuffles: Number of random balanced resamples. The paper uses 1000.
        balance_classes: If ``True``, each shuffle uses equal positives/negatives.
        seed: Seed for the deterministic shuffle generator.

    Returns:
        Tensor of shape ``(batch_size,)`` with AUC per graph.
    """
    if targets.ndim != 3 or preds.ndim != 4:
        raise ValueError("targets must be 3D and preds must be 4D.")

    probs = preds.float().mean(dim=0)
    y_true = targets.reshape(targets.shape[0], -1).float()
    y_score = probs.reshape(probs.shape[0], -1)

    n_shuffles = max(1, int(num_shuffles))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    auc_scores: list[float] = []
    device = targets.device

    for i in range(y_true.shape[0]):
        yt = y_true[i].detach().cpu()
        ys = y_score[i].detach().cpu()

        if not balance_classes:
            auc_scores.append(_auc_from_binary_labels(yt, ys))
            continue

        pos_idx = torch.nonzero(yt > 0.5, as_tuple=False).flatten()
        neg_idx = torch.nonzero(yt <= 0.5, as_tuple=False).flatten()
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            auc_scores.append(0.5)
            continue

        keep = int(min(pos_idx.numel(), neg_idx.numel()))
        auc_samples: list[float] = []
        for _ in range(n_shuffles):
            pos_perm = torch.randperm(pos_idx.numel(), generator=rng)[:keep]
            neg_perm = torch.randperm(neg_idx.numel(), generator=rng)[:keep]
            selected = torch.cat([pos_idx[pos_perm], neg_idx[neg_perm]], dim=0)
            selected = selected[torch.randperm(selected.numel(), generator=rng)]
            auc_samples.append(_auc_from_binary_labels(yt[selected], ys[selected]))

        auc_scores.append(float(sum(auc_samples) / len(auc_samples)))

    return torch.tensor(auc_scores, device=device)


def expected_shd(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Expected SHD for a batch of predictions (Vectorized).

    Args:
        target: (batch_size, num_nodes, num_nodes)
        pred: (num_samples, batch_size, num_nodes, num_nodes)
    Returns:
        Tensor of shape (batch_size,)
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    # pred is (S, B, N, N), target is (B, N, N)
    # Expand target to (1, B, N, N) for broadcasting
    target_expanded = target.unsqueeze(0)

    # SHD (double_for_anticausal=True equivalent) is just sum of absolute differences
    # pred is binary samples, target is binary adjacency
    diff = torch.abs(target_expanded - pred)

    # Sum over N, N dimensions -> (S, B)
    shd_per_sample = diff.sum(dim=(-1, -2))

    # Mean over samples -> (B,)
    return shd_per_sample.float().mean(dim=0)


def threshold_edge_probabilities(edge_probs: torch.Tensor) -> torch.Tensor:
    """Threshold edge probabilities into a single hard graph per batch item."""
    if edge_probs.ndim != 3:
        raise ValueError("edge_probs must be 3D (batch, N, N).")
    return (edge_probs > 0.5).to(dtype=edge_probs.dtype).unsqueeze(0)


def normalized_expected_shd(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Expected SHD normalized by the number of possible directed edges d(d-1).

    This makes SHD comparable across graphs with different node counts.

    Args:
        target: (batch_size, num_nodes, num_nodes)
        pred: (num_samples, batch_size, num_nodes, num_nodes)
    Returns:
        Tensor of shape (batch_size,)
    """
    shd = expected_shd(target, pred)
    n_nodes = target.shape[-1]
    n_possible = n_nodes * (n_nodes - 1)
    if n_possible == 0:
        return torch.zeros_like(shd)
    return shd / n_possible


def expected_f1_score(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Expected F1 score for a batch of predictions (Vectorized).

    When both prediction and target are empty (denom == 0), F1 = 1.0
    (perfect agreement on "no edges"), consistent with ``ancestor_f1_score``
    and ``skeleton_orientation_scores``.

    Args:
        target: (batch_size, num_nodes, num_nodes)
        pred: (num_samples, batch_size, num_nodes, num_nodes)
    Returns:
        Tensor of shape (batch_size,)
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    # Expand target: (1, B, N, N)
    target_expanded = target.unsqueeze(0)

    # Operations on (S, B, N, N)
    tp = (pred * target_expanded).sum(dim=(-1, -2))
    fp = (pred * (1 - target_expanded)).sum(dim=(-1, -2))
    fn = ((1 - pred) * target_expanded).sum(dim=(-1, -2))

    # F1 per sample: (S, B)
    denom = 2 * tp + fp + fn
    # Avoid division by zero
    f1 = torch.where(denom > 0, (2 * tp) / denom, torch.ones_like(denom))

    # Mean over samples -> (B,)
    return f1.mean(dim=0)


def graph_nll_score(
    targets: torch.Tensor,
    preds: torch.Tensor,
    eps: float = 1e-6,
    edge_probs: torch.Tensor | None = None,
) -> float:
    """Negative log-probability of targets under Bernoulli(mean(preds)).

    Uses the posterior mean as the Bernoulli parameter — the Bayes-optimal
    probability estimate and natural choice for calibration evaluation.

    Args:
        targets: ``(batch_size, num_nodes, num_nodes)``
        preds: ``(num_samples, batch_size, num_nodes, num_nodes)``
        eps: Clamp to avoid log(0).

    Returns:
        Scalar mean NLL across the batch.
    """
    targets_t = (
        targets if isinstance(targets, torch.Tensor) else torch.as_tensor(targets)
    )
    preds_t = preds if isinstance(preds, torch.Tensor) else torch.as_tensor(preds)

    p = (
        edge_probs.float() if edge_probs is not None else preds_t.float().mean(dim=0)
    ).clamp(eps, 1.0 - eps)
    y = targets_t.float()
    log_prob = (y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p)).sum(dim=(-1, -2))

    return -float(log_prob.mean().item())


def graph_nll_per_edge_score(
    targets: torch.Tensor,
    preds: torch.Tensor,
    eps: float = 1e-6,
    edge_probs: torch.Tensor | None = None,
) -> float:
    """Negative log-probability averaged per off-diagonal edge.

    Uses the posterior mean as the Bernoulli parameter and masks the diagonal so
    the score is comparable across node counts.

    Args:
        targets: ``(batch_size, num_nodes, num_nodes)``
        preds: ``(num_samples, batch_size, num_nodes, num_nodes)``
        eps: Clamp to avoid log(0).

    Returns:
        Scalar mean NLL per off-diagonal edge across the batch.
    """
    targets_t = (
        targets if isinstance(targets, torch.Tensor) else torch.as_tensor(targets)
    )
    preds_t = preds if isinstance(preds, torch.Tensor) else torch.as_tensor(preds)

    p = (
        edge_probs.float() if edge_probs is not None else preds_t.float().mean(dim=0)
    ).clamp(eps, 1.0 - eps)
    y = targets_t.float()
    n_nodes = y.shape[-1]
    n_possible = n_nodes * (n_nodes - 1)
    if n_possible == 0:
        return 0.0

    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=y.device)
    p_masked = p[:, diag_mask]
    y_masked = y[:, diag_mask]
    log_prob = y_masked * torch.log(p_masked) + (1.0 - y_masked) * torch.log(
        1.0 - p_masked
    )
    nll_per_graph = -log_prob.sum(dim=-1) / float(n_possible)
    return float(nll_per_graph.mean().item())


def edge_entropy(
    preds: torch.Tensor,
    eps: float = 1e-6,
    edge_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean normalized entropy of edge existence probabilities across the batch.

    Edge probabilities are computed as the posterior mean
    ``preds.float().mean(dim=0)``.  Entropy is defined over the
    probability estimate itself, so using the posterior mean is correct
    by definition (it is not a point-estimate approximation).

    Args:
        preds: (num_samples, batch_size, num_nodes, num_nodes)
    Returns:
        Scalar tensor of mean entropy per potential edge, normalized by
        ``log(2)`` so values lie in ``[0, 1]``.
    """
    n_nodes = preds.shape[-1]
    n_possible = n_nodes * (n_nodes - 1)
    if n_possible == 0:
        return torch.tensor(0.0, device=preds.device)

    # Mean probability per edge: (B, N, N)
    p = (
        edge_probs.float() if edge_probs is not None else preds.float().mean(dim=0)
    ).clamp(eps, 1.0 - eps)

    # Binary Entropy: -p log p - (1-p) log (1-p)
    entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))

    # Mask out diagonal (self-loops)
    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=preds.device)
    entropy_masked = entropy[:, diag_mask]

    # Return mean over all potential edges, normalized to [0, 1].
    return entropy_masked.mean() / math.log(2.0)


def structural_interventional_distance(
    target: torch.Tensor, pred: torch.Tensor
) -> torch.Tensor:
    """
    Structural Interventional Distance (SID), averaged over posterior samples.

    Torch-based batched implementation following the SID criterion. The inner
    computation is implemented to run efficiently on GPU via batched tensor
    operations (with a small loop over intervention nodes).

    Args:
        target: (batch_size, num_nodes, num_nodes)
        pred: (num_samples, batch_size, num_nodes, num_nodes)
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    num_samples, batch_size, n_nodes, _ = pred.shape

    # Flatten (S, B) pairs so we can do a single batched SID computation.
    targets_flat = target.unsqueeze(0).expand(num_samples, batch_size, n_nodes, n_nodes)
    targets_flat = targets_flat.reshape(num_samples * batch_size, n_nodes, n_nodes)
    preds_flat = pred.reshape(num_samples * batch_size, n_nodes, n_nodes)

    sid_flat = compute_sid_batched(targets_flat, preds_flat)  # (S*B,)
    sid_sb = sid_flat.view(num_samples, batch_size)

    valid_mask = _valid_dag_sample_mask(target, pred)
    sid_sb = sid_sb.masked_fill(~valid_mask, float("nan"))
    return torch.nanmean(sid_sb, dim=0)


def normalized_structural_interventional_distance(
    target: torch.Tensor, pred: torch.Tensor
) -> torch.Tensor:
    """Expected SID normalized by the maximum possible SID d(d-1).

    This makes SID comparable across graphs with different node counts.

    Args:
        target: (batch_size, num_nodes, num_nodes)
        pred: (num_samples, batch_size, num_nodes, num_nodes)
    Returns:
        Tensor of shape (batch_size,)
    """
    sid = structural_interventional_distance(target, pred)
    n_nodes = target.shape[-1]
    n_possible = n_nodes * (n_nodes - 1)
    if n_possible == 0:
        return torch.zeros_like(sid)
    return sid / n_possible


def compute_sid_batched(adj_true: torch.Tensor, adj_est: torch.Tensor) -> torch.Tensor:
    """
    Compute Structural Interventional Distance (SID) for a batch of graph pairs.

    This follows the SID criterion in a form that is amenable to batched torch ops
    (GPU-friendly). The only Python loops are over intervention nodes (P) and a
    small fixed-point iteration bounded by P.

    Args:
        adj_true: (B, P, P) binary adjacency for true graphs G (edge i -> j is 1 at [i, j]).
        adj_est:  (B, P, P) binary adjacency for estimated graphs H.
    Returns:
        (B,) tensor with SID counts in [0, P*(P-1)].
    """
    if adj_true.ndim != 3 or adj_est.ndim != 3:
        raise ValueError("adj_true and adj_est must be 3D (B, P, P).")
    if adj_true.shape != adj_est.shape or adj_true.shape[-1] != adj_true.shape[-2]:
        raise ValueError(
            "adj_true and adj_est must have the same square shape (B, P, P)."
        )

    bsz, p, _ = adj_true.shape
    if p <= 1:
        return torch.zeros(bsz, device=adj_true.device, dtype=torch.float32)

    device = adj_true.device

    # Ensure float adjacency for matmul, but treat as boolean masks.
    g = (adj_true > 0).to(dtype=torch.float32)
    h = (adj_est > 0).to(dtype=torch.float32)

    # 1) Precompute transitive closure / reachability for G.
    # T_G[b, i, j] is True iff i is an ancestor of j in G (path length >= 1).
    t_g = _transitive_closure_bool(g)  # (B, P, P) bool
    t_g_f = t_g.to(dtype=torch.float32)

    # Ancestors in G (including self): Anc[b, i, j] True iff j is ancestor of i or j==i.
    eye = torch.eye(p, device=device, dtype=torch.bool).unsqueeze(0).expand(bsz, p, p)
    anc = t_g.transpose(1, 2) | eye

    sid_counts = torch.zeros(bsz, device=device, dtype=torch.float32)

    for i in range(p):
        # Z = Parents of i in H (column i of H).
        z_mask = h[:, :, i] > 0  # (B, P) bool

        # Check 0: reverse edges (a parent of i in H is a descendant of i in G).
        err_reverse = z_mask & t_g[:, i, :]  # (B, P) bool

        # Check 1: Condition (*) part 1 (directed paths through "bad mediators").
        w_is_anc_z = (
            torch.matmul(t_g_f, z_mask.unsqueeze(-1).to(torch.float32)).squeeze(-1)
        ) > 0
        w_is_desc_i = t_g[:, i, :]  # descendants of i in G
        bad_w = w_is_desc_i & w_is_anc_z  # nodes on a directed path from i that reach Z
        err_cond1 = (
            torch.matmul(bad_w.unsqueeze(1).to(torch.float32), t_g_f).squeeze(1)
        ) > 0

        # Check 2: Condition (*) part 2 (non-directed paths; d-connection in G_{underline{i}}).
        g_back = g.clone()
        g_back[:, i, :] = 0.0  # remove edges leaving i

        # Nodes that are in Anc(Z) (including Z itself) for each batch item.
        z_anc_mask = (
            torch.matmul(
                anc.to(torch.float32), z_mask.unsqueeze(-1).to(torch.float32)
            ).squeeze(-1)
        ) > 0

        # Batched Bayes-ball reachability (fixed-point iteration).
        curr_up = torch.zeros(bsz, p, device=device, dtype=torch.bool)
        curr_down = torch.zeros(bsz, p, device=device, dtype=torch.bool)
        curr_up[:, i] = True

        for _ in range(p):
            not_z = ~z_mask

            can_pass_down = (curr_down | curr_up) & not_z
            next_down = (
                torch.matmul(
                    can_pass_down.to(torch.float32).unsqueeze(1), g_back
                ).squeeze(1)
            ) > 0

            can_pass_up_chain = curr_up & not_z
            can_pass_up_collider = curr_down & z_anc_mask
            can_pass_up = can_pass_up_chain | can_pass_up_collider
            next_up = (
                torch.matmul(
                    can_pass_up.to(torch.float32).unsqueeze(1), g_back.transpose(1, 2)
                ).squeeze(1)
            ) > 0

            new_down = curr_down | next_down
            new_up = curr_up | next_up
            if torch.equal(new_down, curr_down) and torch.equal(new_up, curr_up):
                break
            curr_down = new_down
            curr_up = new_up

        d_connected = curr_down | curr_up
        d_connected[:, i] = False

        # Combine errors for node i.
        in_z = z_mask
        errors_in_z = in_z & err_reverse
        errors_out_z = (~in_z) & (err_cond1 | d_connected)

        total_errors = errors_in_z | errors_out_z
        total_errors[:, i] = False
        sid_counts += total_errors.sum(dim=1).to(torch.float32)

    return sid_counts


def _transitive_closure_bool(adj: torch.Tensor) -> torch.Tensor:
    """
    Compute reachability (transitive closure) for a batch of adjacency matrices.

    Args:
        adj: (..., N, N) adjacency (0/1 or bool)
    Returns:
        (..., N, N) bool reachability matrix (excluding diagonal)
    """
    if adj.ndim < 2:
        raise ValueError("adj must have at least 2 dims.")
    if adj.shape[-1] != adj.shape[-2]:
        raise ValueError("adj must be square in the last two dims.")

    n = int(adj.shape[-1])
    reach = adj.to(dtype=torch.bool)

    if n <= 1:
        return reach

    # Repeated squaring for transitive closure:
    # reach <- reach OR (reach @ reach), repeated log2(n) times is sufficient.
    steps = int(math.ceil(math.log2(n)))
    reach_f = reach.to(dtype=torch.float32)
    for _ in range(steps):
        reach = reach | (reach_f @ reach_f > 0)
        reach_f = reach.to(dtype=torch.float32)

    eye = torch.eye(n, device=adj.device, dtype=torch.bool)
    return reach & ~eye


def ancestor_f1_score(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """F1 score for causal ancestor relationships.

    Checks if i is an ancestor of j in both graphs.
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    # Compute reachability for target once: (B, N, N)
    tgt_reach = _transitive_closure_bool(target)

    # Compute reachability for all predicted samples: (S, B, N, N)
    pred_reach = _transitive_closure_bool(pred)

    # F1 per sample and batch: (S, B)
    tp = (pred_reach & tgt_reach.unsqueeze(0)).sum(dim=(-1, -2)).to(dtype=torch.float32)
    fp = (
        (pred_reach & ~tgt_reach.unsqueeze(0)).sum(dim=(-1, -2)).to(dtype=torch.float32)
    )
    fn = (
        ((~pred_reach) & tgt_reach.unsqueeze(0))
        .sum(dim=(-1, -2))
        .to(dtype=torch.float32)
    )

    denom = 2.0 * tp + fp + fn
    f1 = torch.where(denom > 0, (2.0 * tp) / denom, torch.ones_like(denom))
    return f1.mean(dim=0)


class Metrics(BaseMetrics):
    """Metrics handler for graph structure learning (probabilistic only).

    Tracks metrics for a **single** dataset (SCM family).  Callers must
    create one instance per family or call ``reset()`` between families.

    Workflow::

        m = Metrics(metrics=["e-shd", "auc"])
        for batch in loader:
            m.update(targets, samples)
        summary = m.compute(summary_stats=True)   # DDP-safe
        raw     = m.gather_raw_results()           # for persistence
    """

    def __init__(
        self,
        metrics: List[str] | None = None,
        *,
        auc_num_shuffles: int = 1000,
        auc_balance_classes: bool = True,
        auc_seed: int = 0,
    ):
        super().__init__()
        self.metrics_list = (
            metrics
            if metrics is not None
            else [
                "e-shd",
                "e-edgef1",
                "e-sid",
                "ne-shd",
                "ne-sid",
                "graph_nll",
                "graph_nll_per_edge",
                "edge_entropy",
                "ancestor_f1",
                "auc",
                "sparsity_ratio",
                "skeleton_f1",
                "orientation_accuracy",
                "valid_dag_pct",
                "sampled-e-shd",
                "sampled-e-edgef1",
                "sampled-e-sid",
                "sampled-ne-shd",
                "sampled-ne-sid",
                "sampled-graph_nll",
                "sampled-graph_nll_per_edge",
                "sampled-edge_entropy",
                "sampled-ece",
                "sampled-ancestor_f1",
                "sampled-sparsity_ratio",
                "sampled-skeleton_f1",
                "sampled-orientation_accuracy",
                "sampled-valid_dag_pct",
                "threshold_valid_dag_pct",
                "ece",
            ]
        )
        self.auc_num_shuffles = max(1, int(auc_num_shuffles))
        self.auc_balance_classes = bool(auc_balance_classes)
        self.auc_seed = int(auc_seed)

    def _compute_batch_metrics(
        self,
        targets: torch.Tensor,
        samples: torch.Tensor,
        *,
        thresholded_graph: torch.Tensor | None = None,
        edge_probs: torch.Tensor | None = None,
    ) -> Dict[str, float]:
        """Compute the configured metrics for a single batch.

        Args:
            targets: ``(B, N, N)`` ground truth adjacency.
            samples: ``(S, B, N, N)`` sampled graphs.
        """
        batch_metrics: Dict[str, float] = {}
        structural_pred = (
            thresholded_graph if thresholded_graph is not None else samples
        )

        if "e-shd" in self.metrics_list:
            batch_metrics["e-shd"] = float(
                expected_shd(targets, structural_pred).mean().item()
            )

        if "ne-shd" in self.metrics_list:
            batch_metrics["ne-shd"] = float(
                normalized_expected_shd(targets, structural_pred).mean().item()
            )

        if "e-edgef1" in self.metrics_list:
            batch_metrics["e-edgef1"] = float(
                expected_f1_score(targets, structural_pred).mean().item()
            )

        if "ancestor_f1" in self.metrics_list:
            batch_metrics["ancestor_f1"] = float(
                ancestor_f1_score(targets, structural_pred).mean().item()
            )

        if "e-sid" in self.metrics_list:
            batch_metrics["e-sid"] = float(
                structural_interventional_distance(targets, structural_pred)
                .mean()
                .item()
            )

        if "ne-sid" in self.metrics_list:
            batch_metrics["ne-sid"] = float(
                normalized_structural_interventional_distance(targets, structural_pred)
                .mean()
                .item()
            )

        if "graph_nll" in self.metrics_list:
            batch_metrics["graph_nll"] = graph_nll_score(
                targets, samples, edge_probs=edge_probs
            )

        if "graph_nll_per_edge" in self.metrics_list:
            batch_metrics["graph_nll_per_edge"] = graph_nll_per_edge_score(
                targets, samples, edge_probs=edge_probs
            )

        if "edge_entropy" in self.metrics_list:
            batch_metrics["edge_entropy"] = float(
                edge_entropy(samples, edge_probs=edge_probs).item()
            )

        if "auc" in self.metrics_list:
            batch_metrics["auc"] = float(
                auc_graph_scores_configurable(
                    targets,
                    samples,
                    num_shuffles=self.auc_num_shuffles,
                    balance_classes=self.auc_balance_classes,
                    seed=self.auc_seed,
                )
                .mean()
                .item()
            )

        # ── Edge confusion decomposition ───────────────────────────────
        confusion_keys = {"fp_count", "fn_count", "reversed_count", "correct_count"}
        if confusion_keys & set(self.metrics_list):
            decomp = edge_confusion_decomposition(targets, samples)
            for short, full in [
                ("fp", "fp_count"),
                ("fn", "fn_count"),
                ("reversed", "reversed_count"),
                ("correct", "correct_count"),
            ]:
                if full in self.metrics_list:
                    batch_metrics[full] = float(decomp[short].mean().item())

        # ── Sparsity ratio ─────────────────────────────────────────────
        if "sparsity_ratio" in self.metrics_list:
            batch_metrics["sparsity_ratio"] = float(
                sparsity_ratio(targets, structural_pred).mean().item()
            )

        # ── Skeleton / orientation split ───────────────────────────────
        skel_keys = {"skeleton_f1", "orientation_accuracy"}
        if skel_keys & set(self.metrics_list):
            skel = skeleton_orientation_scores(targets, structural_pred)
            if "skeleton_f1" in self.metrics_list:
                batch_metrics["skeleton_f1"] = float(skel["skeleton_f1"].mean().item())
            if "orientation_accuracy" in self.metrics_list:
                batch_metrics["orientation_accuracy"] = float(
                    skel["orientation_accuracy"].mean().item()
                )

        # ── Expected Calibration Error ─────────────────────────────────
        if "ece" in self.metrics_list:
            batch_metrics["ece"] = float(
                expected_calibration_error(
                    targets, samples, edge_probs=edge_probs
                ).item()
            )

        if "valid_dag_pct" in self.metrics_list:
            batch_metrics["valid_dag_pct"] = float(
                100.0
                * (
                    threshold_valid_dag_rate_from_probs(targets, edge_probs)
                    .mean()
                    .item()
                    if edge_probs is not None
                    else valid_dag_rate(targets, structural_pred).mean().item()
                )
            )

        if "sampled-e-shd" in self.metrics_list:
            batch_metrics["sampled-e-shd"] = float(
                expected_shd(targets, samples).mean().item()
            )

        if "sampled-ne-shd" in self.metrics_list:
            batch_metrics["sampled-ne-shd"] = float(
                normalized_expected_shd(targets, samples).mean().item()
            )

        if "sampled-e-edgef1" in self.metrics_list:
            batch_metrics["sampled-e-edgef1"] = float(
                expected_f1_score(targets, samples).mean().item()
            )

        if "sampled-ancestor_f1" in self.metrics_list:
            batch_metrics["sampled-ancestor_f1"] = float(
                ancestor_f1_score(targets, samples).mean().item()
            )

        if "sampled-e-sid" in self.metrics_list:
            batch_metrics["sampled-e-sid"] = float(
                structural_interventional_distance(targets, samples).mean().item()
            )

        if "sampled-ne-sid" in self.metrics_list:
            batch_metrics["sampled-ne-sid"] = float(
                normalized_structural_interventional_distance(targets, samples)
                .mean()
                .item()
            )

        if "sampled-graph_nll" in self.metrics_list:
            batch_metrics["sampled-graph_nll"] = graph_nll_score(targets, samples)

        if "sampled-graph_nll_per_edge" in self.metrics_list:
            batch_metrics["sampled-graph_nll_per_edge"] = graph_nll_per_edge_score(
                targets, samples
            )

        if "sampled-edge_entropy" in self.metrics_list:
            batch_metrics["sampled-edge_entropy"] = float(edge_entropy(samples).item())

        if "sampled-ece" in self.metrics_list:
            batch_metrics["sampled-ece"] = float(
                expected_calibration_error(targets, samples).item()
            )

        if "sampled-sparsity_ratio" in self.metrics_list:
            batch_metrics["sampled-sparsity_ratio"] = float(
                sparsity_ratio(targets, samples).mean().item()
            )

        sampled_skel_keys = {"sampled-skeleton_f1", "sampled-orientation_accuracy"}
        if sampled_skel_keys & set(self.metrics_list):
            sampled_skel = skeleton_orientation_scores(targets, samples)
            if "sampled-skeleton_f1" in self.metrics_list:
                batch_metrics["sampled-skeleton_f1"] = float(
                    sampled_skel["skeleton_f1"].mean().item()
                )
            if "sampled-orientation_accuracy" in self.metrics_list:
                batch_metrics["sampled-orientation_accuracy"] = float(
                    sampled_skel["orientation_accuracy"].mean().item()
                )

        if "sampled-valid_dag_pct" in self.metrics_list:
            batch_metrics["sampled-valid_dag_pct"] = float(
                100.0 * valid_dag_rate(targets, samples).mean().item()
            )

        if "threshold_valid_dag_pct" in self.metrics_list:
            batch_metrics["threshold_valid_dag_pct"] = float(
                100.0
                * (
                    threshold_valid_dag_rate_from_probs(targets, edge_probs)
                    .mean()
                    .item()
                    if edge_probs is not None
                    else threshold_valid_dag_rate(targets, samples).mean().item()
                )
            )

        return batch_metrics

    def update(
        self,
        targets: torch.Tensor,
        samples: torch.Tensor,
        *,
        thresholded_graph: torch.Tensor | None = None,
        edge_probs: torch.Tensor | None = None,
    ) -> None:
        """Compute metrics for a batch and append to internal history.

        Args:
            targets: ``(B, N, N)`` ground truth adjacency.
            samples: ``(S, B, N, N)`` binary graph samples.
        """
        batch_metrics = self._compute_batch_metrics(
            targets,
            samples,
            thresholded_graph=thresholded_graph,
            edge_probs=edge_probs,
        )
        for k, v in batch_metrics.items():
            self.history[k].append(v)

    def compute(self, *, summary_stats: bool = True) -> Dict[str, Any]:
        """Summarise accumulated history with DDP-safe aggregation.

        Must be called after one or more ``update()`` calls.

        Args:
            summary_stats: If ``True``, returns ``{metric}_mean/_sem/_std``.
                Otherwise returns ``{metric}: mean`` only.
        """
        return self.summarize(summary_stats=summary_stats)


def edge_confusion_decomposition(
    target: torch.Tensor, pred: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Decompose directed-edge prediction errors into FP, FN, reversed, and correct.

    For each posterior sample, edges are classified as:
    - **correct**: ``target[i,j]=1`` and ``pred[i,j]=1``
    - **reversed**: ``target[i,j]=1`` and ``pred[j,i]=1`` (but ``pred[i,j]=0``)
    - **fn** (false negative): ``target[i,j]=1`` but neither ``pred[i,j]`` nor ``pred[j,i]``
    - **fp** (false positive): ``pred[i,j]=1`` but ``target[i,j]=0`` (and not counted as reversed)

    Counts are averaged over posterior samples, yielding per-graph expected counts.

    Args:
        target: (B, N, N) ground truth binary adjacency.
        pred: (S, B, N, N) sampled binary adjacency.

    Returns:
        Dict with keys ``"fp"``, ``"fn"``, ``"reversed"``, ``"correct"``, each
        a tensor of shape ``(B,)`` with expected counts per graph.
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    # Expand target: (1, B, N, N) for broadcasting against (S, B, N, N)
    t = target.unsqueeze(0).float()  # (1, B, N, N)
    p = pred.float()  # (S, B, N, N)

    # Correct: true edge present in prediction with correct direction
    correct = (t * p).sum(dim=(-1, -2))  # (S, B)

    # Reversed: target[i,j]=1 and pred[j,i]=1 and pred[i,j]=0
    p_transposed = p.transpose(-1, -2)  # (S, B, N, N)
    reversed_edges = (t * p_transposed * (1 - p)).sum(dim=(-1, -2))  # (S, B)

    # False negatives: target[i,j]=1 but pred[i,j]=0 and pred[j,i]=0
    fn = (t * (1 - p) * (1 - p_transposed)).sum(dim=(-1, -2))  # (S, B)

    # False positives: pred[i,j]=1 but target[i,j]=0.
    # Exclude only "pure reversed" cases (target[j,i]=1 and pred[j,i]=0),
    # since those are already captured by `reversed_edges` and contribute 2
    # units to SHD via the decomposition fp + fn + 2*reversed.
    # If both directions are predicted while only one is true, the extra
    # direction is a genuine FP and must be counted here.
    t_transposed = target.unsqueeze(0).float().transpose(-1, -2)
    pure_reversed_mask = t_transposed * (1 - p_transposed)
    fp = (p * (1 - t) * (1 - pure_reversed_mask)).sum(dim=(-1, -2))  # (S, B)

    # Average over posterior samples → expected counts per graph
    return {
        "fp": fp.mean(dim=0),
        "fn": fn.mean(dim=0),
        "reversed": reversed_edges.mean(dim=0),
        "correct": correct.mean(dim=0),
    }


def sparsity_ratio(
    target: torch.Tensor, pred: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Ratio of predicted density to true density, averaged over posterior samples.

    Values < 1 indicate under-prediction (sparse outputs), > 1 indicate
    over-prediction (dense outputs).

    Args:
        target: (B, N, N) ground truth binary adjacency.
        pred: (S, B, N, N) sampled binary adjacency.
        eps: Small constant to avoid division by zero for empty true graphs.

    Returns:
        Tensor of shape ``(B,)`` with per-graph sparsity ratios.
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")

    n_nodes = target.shape[-1]
    n_potential = n_nodes * (n_nodes - 1)  # off-diagonal entries

    true_density = target.float().sum(dim=(-1, -2)) / max(n_potential, 1)  # (B,)
    pred_density = pred.float().sum(dim=(-1, -2)) / max(n_potential, 1)  # (S, B)
    pred_density_mean = pred_density.mean(dim=0)  # (B,)

    return pred_density_mean / (true_density + eps)


def skeleton_orientation_scores(
    target: torch.Tensor, pred: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Skeleton F1 and orientation accuracy for directed graphs.

    The *skeleton* ignores edge direction: an edge exists between ``(i, j)`` if
    ``adj[i,j]=1 OR adj[j,i]=1``.  Skeleton F1 measures how well the model
    recovers the undirected structure.

    *Orientation accuracy* is computed **only** over edges present in both the
    true and predicted skeletons.  It measures the fraction of those skeleton
    edges whose direction is correct.

    Both are averaged over posterior samples.

    Args:
        target: (B, N, N) ground truth binary adjacency.
        pred: (S, B, N, N) sampled binary adjacency.

    Returns:
        Dict with ``"skeleton_f1"`` (B,) and ``"orientation_accuracy"`` (B,).
        Orientation accuracy is 1.0 when there are no common skeleton edges.
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    t = target.float()
    # True skeleton: (B, N, N) — upper triangle only to avoid double counting
    t_skel = torch.clamp(t + t.transpose(-1, -2), max=1.0)

    # Extract upper triangle mask
    idx = torch.triu_indices(t.shape[-2], t.shape[-1], offset=1, device=t.device)

    # True skeleton upper-tri: (B, E)
    t_skel_ut = t_skel[:, idx[0], idx[1]]  # (B, E)

    num_samples = pred.shape[0]
    skeleton_f1_accum = torch.zeros(target.shape[0], device=target.device)
    orient_acc_accum = torch.zeros(target.shape[0], device=target.device)

    for s in range(num_samples):
        p = pred[s].float()  # (B, N, N)
        p_skel = torch.clamp(p + p.transpose(-1, -2), max=1.0)
        p_skel_ut = p_skel[:, idx[0], idx[1]]  # (B, E)

        # Skeleton F1
        tp = (p_skel_ut * t_skel_ut).sum(dim=-1)  # (B,)
        fp = (p_skel_ut * (1 - t_skel_ut)).sum(dim=-1)
        fn = ((1 - p_skel_ut) * t_skel_ut).sum(dim=-1)
        denom = 2 * tp + fp + fn
        f1 = torch.where(denom > 0, (2 * tp) / denom, torch.ones_like(denom))
        skeleton_f1_accum += f1

        # Orientation accuracy: among common skeleton edges, how many have
        # correct direction?  For each upper-tri pair (i,j) in both skeletons,
        # check whether the actual directed edge matches.
        common = p_skel_ut * t_skel_ut  # (B, E) — 1 where both skeletons agree

        # Direction match: for pair (i,j), direction is correct if
        # target[i,j]==pred[i,j] AND target[j,i]==pred[j,i]
        dir_match = (t[:, idx[0], idx[1]] == p[:, idx[0], idx[1]]).float() * (
            t[:, idx[1], idx[0]] == p[:, idx[1], idx[0]]
        ).float()  # (B, E)

        correct_dir = (common * dir_match).sum(dim=-1)  # (B,)
        total_common = common.sum(dim=-1)  # (B,)
        orient_acc = torch.where(
            total_common > 0,
            correct_dir / total_common,
            torch.ones_like(total_common),
        )
        orient_acc_accum += orient_acc

    return {
        "skeleton_f1": skeleton_f1_accum / num_samples,
        "orientation_accuracy": orient_acc_accum / num_samples,
    }


def expected_calibration_error(
    target: torch.Tensor,
    pred: torch.Tensor,
    n_bins: int = 10,
    edge_probs: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expected Calibration Error (ECE) for edge-existence probabilities.

    Edge probabilities are obtained as ``pred.float().mean(dim=0)`` (the
    posterior mean over samples).  The posterior mean is the natural
    Bayes-optimal probability estimate, making it the correct input for
    calibration evaluation.  Probabilities are binned into ``n_bins``
    equal-width bins over [0, 1].  ECE is the weighted average of
    ``|accuracy_bin - confidence_bin|`` across bins.

    Args:
        target: (B, N, N) ground truth binary adjacency.
        pred: (S, B, N, N) sampled binary adjacency.
        n_bins: Number of equal-width bins (default 10).

    Returns:
        Scalar tensor with the batch-averaged ECE.
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")

    # Mean edge probability from posterior samples: (B, N, N)
    prob = edge_probs.float() if edge_probs is not None else pred.float().mean(dim=0)

    # Mask out diagonal (self-loop) entries — these are always 0/0 in DAGs
    # and would bias ECE downward by giving the model easy credit.
    n_nodes = target.shape[-1]
    diag_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=target.device)
    prob_masked = prob[:, diag_mask]  # (B, N*(N-1))
    target_masked = target.float()[:, diag_mask]  # (B, N*(N-1))

    prob_flat = prob_masked
    target_flat = target_masked

    batch_size = prob_flat.shape[0]
    ece_per_graph = torch.zeros(batch_size, device=target.device)

    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1, device=target.device)

    for b in range(batch_size):
        p = prob_flat[b]
        y = target_flat[b]
        total = p.numel()

        ece = torch.tensor(0.0, device=target.device)
        for i in range(n_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            if i == n_bins - 1:
                in_bin = (p >= lo) & (p <= hi)
            else:
                in_bin = (p >= lo) & (p < hi)

            n_in_bin = in_bin.sum().float()
            if n_in_bin < 1:
                continue

            avg_confidence = p[in_bin].mean()
            avg_accuracy = y[in_bin].mean()
            ece += (n_in_bin / total) * torch.abs(avg_accuracy - avg_confidence)

        ece_per_graph[b] = ece

    return ece_per_graph.mean()


def _valid_dag_sample_mask(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Return a boolean DAG-validity mask for posterior graph samples."""
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    num_samples, batch_size, n_nodes, _ = pred.shape
    pred_flat = (pred > 0.5).reshape(num_samples * batch_size, n_nodes, n_nodes)

    # Compute transitive closure without masking diagonal; a diagonal True entry
    # indicates a directed cycle (including explicit self-loops).
    reach = pred_flat
    if n_nodes > 1:
        steps = int(math.ceil(math.log2(n_nodes)))
        reach_f = reach.to(dtype=torch.float32)
        for _ in range(steps):
            reach = reach | (reach_f @ reach_f > 0)
            reach_f = reach.to(dtype=torch.float32)

    has_cycle = torch.diagonal(reach, dim1=-2, dim2=-1).any(dim=-1)
    return (~has_cycle).view(num_samples, batch_size)


def valid_dag_rate(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Fraction of posterior graph samples that are valid DAGs.

    Args:
        target: (B, N, N) ground truth binary adjacency.
        pred: (S, B, N, N) sampled binary adjacency.

    Returns:
        Tensor of shape ``(B,)`` with per-task fractions in ``[0, 1]``.
    """
    return _valid_dag_sample_mask(target, pred).to(dtype=torch.float32).mean(dim=0)


def threshold_valid_dag_rate(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Validity of the thresholded posterior-mean graph.

    This mirrors AVICI's reported use of a single hard prediction graph obtained
    from posterior edge probabilities, rather than posterior graph samples.

    Args:
        target: ``(B, N, N)`` ground truth binary adjacency.
        pred: ``(S, B, N, N)`` sampled binary adjacency.

    Returns:
        Tensor of shape ``(B,)`` containing ``1`` if the thresholded mean graph
        is a DAG and ``0`` otherwise.
    """
    if target.ndim != 3 or pred.ndim != 4:
        raise ValueError("target must be 3D and pred must be 4D.")
    if pred.shape[1:] != target.shape:
        raise ValueError(
            "pred shape must be (num_samples, batch, N, N) matching target."
        )

    mean_graph = (pred.float().mean(dim=0) > 0.5).to(dtype=pred.dtype)
    mean_graph = mean_graph.unsqueeze(0)
    return valid_dag_rate(target, mean_graph)


def threshold_valid_dag_rate_from_probs(
    target: torch.Tensor, edge_probs: torch.Tensor
) -> torch.Tensor:
    """Validity of the thresholded edge-probability graph."""
    return valid_dag_rate(target, threshold_edge_probabilities(edge_probs))
