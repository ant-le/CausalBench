from __future__ import annotations

from typing import Any, Optional

import torch

from causal_meta.models.base import BaseModel
from causal_meta.models.factory import register_model


@register_model("random")
class RandomModel(BaseModel):
    """Random DAG baseline with training-sparsity-matched edge sampling.

    The model samples edges independently from a Bernoulli distribution with
    probability ``p_edge`` and enforces acyclicity by first sampling an
    upper-triangular adjacency matrix and then applying a random node permutation.

    Args:
        num_nodes: Number of variables in each graph.
        p_edge: Bernoulli edge probability. This should be matched to the
            expected sparsity of the training graph distribution.
        randomize_topological_order: If ``True``, applies a random permutation to
            each sampled graph so the baseline does not depend on canonical node order.
    """

    def __init__(
        self,
        num_nodes: int,
        p_edge: float,
        randomize_topological_order: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        _ = kwargs

        if num_nodes < 1:
            raise ValueError("num_nodes must be >= 1.")
        if not (0.0 <= float(p_edge) <= 1.0):
            raise ValueError("p_edge must be in [0, 1].")

        self.num_nodes = int(num_nodes)
        self.p_edge = float(p_edge)
        self.randomize_topological_order = bool(randomize_topological_order)

    def set_num_nodes(self, num_nodes: int) -> None:
        """Update the active node count for the next explicit inference call."""
        resolved = int(num_nodes)
        if resolved <= 0:
            raise ValueError("num_nodes must be positive.")
        self.num_nodes = resolved

    @property
    def needs_pretraining(self) -> bool:
        """Random baseline has no trainable inference procedure."""
        return False

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """Forward is unsupported for explicit sampling baselines.

        Args:
            x: Input data tensor.
            mask: Unused — accepted for interface compatibility.

        Raises:
            RuntimeError: Always raised, use :meth:`sample` instead.
        """
        _ = x
        _ = mask
        raise RuntimeError("RandomModel does not implement forward(); use sample().")

    def sample(
        self,
        x: torch.Tensor,
        num_samples: int = 1,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample random DAGs.

        Args:
            x: Input data of shape ``(batch_size, n_observations, n_nodes)``.
            num_samples: Number of graph samples per task.
            mask: Unused — accepted for interface compatibility.

        Returns:
            Tensor of sampled graphs with shape
            ``(batch_size, num_samples, n_nodes, n_nodes)``.
        """
        if x.ndim != 3:
            raise ValueError("Input data must have shape (batch, samples, nodes).")

        batch_size = int(x.shape[0])
        n_nodes = int(x.shape[-1])
        if n_nodes != self.num_nodes:
            raise ValueError(
                "Input data node count does not match configured num_nodes."
            )

        device = x.device
        upper_mask = torch.triu(
            torch.ones((n_nodes, n_nodes), device=device, dtype=torch.bool), diagonal=1
        )

        base = torch.rand(
            (batch_size, num_samples, n_nodes, n_nodes),
            device=device,
            dtype=torch.float32,
        )
        graphs = (base < self.p_edge) & upper_mask.view(1, 1, n_nodes, n_nodes)
        graphs = graphs.to(dtype=torch.float32)

        if not self.randomize_topological_order:
            return graphs

        perms = torch.stack(
            [
                torch.randperm(n_nodes, device=device)
                for _ in range(batch_size * int(num_samples))
            ],
            dim=0,
        ).view(batch_size, int(num_samples), n_nodes)

        row_idx = perms.unsqueeze(-1).expand(
            batch_size, int(num_samples), n_nodes, n_nodes
        )
        graphs = torch.gather(graphs, dim=2, index=row_idx)
        col_idx = perms.unsqueeze(-2).expand(
            batch_size, int(num_samples), n_nodes, n_nodes
        )
        graphs = torch.gather(graphs, dim=3, index=col_idx)
        return graphs

    def calculate_loss(
        self,
        output: Any,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Loss is undefined for random baseline.

        Args:
            output: Unused.
            target: Unused.

        Raises:
            RuntimeError: Always raised.
        """
        _ = output
        _ = target
        _ = kwargs
        raise RuntimeError("RandomModel does not implement calculate_loss().")
