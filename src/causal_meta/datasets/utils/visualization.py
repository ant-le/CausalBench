from __future__ import annotations

from typing import TYPE_CHECKING, List

import numpy as np
import torch

from causal_meta.datasets.utils.stats import degree_sequence

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from causal_meta.datasets.scm import SCMFamily


def plot_degree_distribution(family: SCMFamily, n_samples: int = 100) -> Figure:
    """Plot a histogram of node degrees sampled from a family."""
    import matplotlib.pyplot as plt

    if n_samples < 1:
        raise ValueError("n_samples must be positive.")

    degrees: List[float] = []
    for seed in range(n_samples):
        adj = family.sample_task(seed).adjacency_matrix.detach().cpu()
        degrees.extend(degree_sequence(adj).tolist())

    fig, ax = plt.subplots()
    if degrees:
        max_degree = max(1, int(max(degrees)))
        bins = np.arange(0, max_degree + 2) - 0.5
        ax.hist(degrees, bins=bins, edgecolor="black")
        ax.set_xticks(range(0, max_degree + 1))
    else:
        ax.hist([])

    ax.set_xlabel("Node Degree")
    ax.set_ylabel("Frequency")
    ax.set_title("Degree Distribution")
    fig.tight_layout()
    return fig


def visualize_adjacency(adj_tensor: torch.Tensor) -> Figure:
    """Visualize a single adjacency matrix."""
    import matplotlib.pyplot as plt

    array = adj_tensor.detach().cpu().numpy()
    fig, ax = plt.subplots()
    im = ax.imshow(array, cmap="Blues", interpolation="nearest")
    ax.set_xlabel("Child Node")
    ax.set_ylabel("Parent Node")
    ax.set_title("Adjacency Matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
