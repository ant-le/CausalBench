"""SynTReN synthetic gene regulatory network benchmark.

SynTReN (Van den Bulcke et al., 2006) generates synthetic transcriptional
regulatory networks based on known sub-networks of *E. coli* and *S. cerevisiae*.
The BayesDAG paper (Annadani et al., 2023) uses a d=20 SynTReN network with
E-SHD=34.21 as a real-world benchmark.

Unlike Sachs, SynTReN datasets are generated per-run and there is no single
canonical download URL.  This loader therefore expects the user to provide a
local directory containing:

- ``data.csv`` or ``data.tsv``: expression matrix (rows=samples, cols=genes)
- ``adjacency.csv`` or ``adjacency.tsv``: ground-truth adjacency matrix (d × d)

Alternatively, a single ``.npz`` file containing ``data`` and ``adjacency``
arrays is also accepted.

Reference:
    Van den Bulcke, T., Van Leemput, K., Naudts, B., van de Peer, Y.,
    Laukens, K., Podesta, A., ... & Marchal, K. (2006).  SynTReN: a generator
    of synthetic gene expression data for design and analysis of structure
    learning algorithms.  *BMC Bioinformatics*, 7(1), 43.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from causal_meta.datasets.real_world.registry import _register

log = logging.getLogger(__name__)

# Default data directory (can be overridden via loader_kwargs).
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "_cache" / "syntren"


def _load_matrix_file(path: Path) -> np.ndarray:
    """Load a CSV or TSV matrix file, auto-detecting the delimiter."""
    with open(path) as f:
        first_line = f.readline()

    delimiter = "\t" if "\t" in first_line else ","

    # Try to detect whether a header row is present (non-numeric first cell).
    parts = first_line.strip().split(delimiter)
    has_header = False
    try:
        float(parts[0])
    except ValueError:
        has_header = True

    return np.loadtxt(
        path,
        delimiter=delimiter,
        skiprows=1 if has_header else 0,
        dtype=np.float32,
    )


def load_syntren(
    *,
    data_dir: str | None = None,
    **kwargs: object,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a SynTReN dataset from a local directory.

    Args:
        data_dir: Path to the directory containing data and adjacency files.
            Defaults to ``<package>/_cache/syntren/``.

    Returns:
        ``(data, adjacency)`` — both float32 tensors.

    Raises:
        FileNotFoundError: If the data directory or required files are missing.
    """
    root = Path(data_dir) if data_dir is not None else _DEFAULT_DATA_DIR

    if not root.exists():
        raise FileNotFoundError(
            f"SynTReN data directory not found: {root}\n"
            "Please place the SynTReN benchmark files in this directory.\n"
            "Expected files: data.csv (or .tsv) and adjacency.csv (or .tsv),\n"
            "or a single data.npz with 'data' and 'adjacency' arrays."
        )

    # Try .npz first.
    npz_path = root / "data.npz"
    if npz_path.exists():
        log.info("Loading SynTReN dataset from %s", npz_path)
        loaded = np.load(npz_path)
        data = torch.from_numpy(loaded["data"].astype(np.float32))
        adjacency = torch.from_numpy(loaded["adjacency"].astype(np.float32))
        log.info(
            "SynTReN: %d samples, %d variables, %d edges",
            data.shape[0],
            data.shape[1],
            int(adjacency.sum().item()),
        )
        return data, adjacency

    # Try CSV/TSV files.
    data_path = None
    for suffix in ("csv", "tsv", "txt"):
        candidate = root / f"data.{suffix}"
        if candidate.exists():
            data_path = candidate
            break

    adj_path = None
    for suffix in ("csv", "tsv", "txt"):
        candidate = root / f"adjacency.{suffix}"
        if candidate.exists():
            adj_path = candidate
            break

    if data_path is None:
        raise FileNotFoundError(
            f"No data file found in {root}. "
            "Expected data.csv, data.tsv, data.txt, or data.npz."
        )
    if adj_path is None:
        raise FileNotFoundError(
            f"No adjacency file found in {root}. "
            "Expected adjacency.csv, adjacency.tsv, or adjacency.txt."
        )

    log.info("Loading SynTReN data from %s and %s", data_path, adj_path)
    data = torch.from_numpy(_load_matrix_file(data_path))
    adjacency = torch.from_numpy(_load_matrix_file(adj_path))

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {adjacency.shape}")
    if data.shape[1] != adjacency.shape[0]:
        raise ValueError(
            f"Data has {data.shape[1]} variables but adjacency is "
            f"{adjacency.shape[0]}×{adjacency.shape[1]}"
        )

    log.info(
        "SynTReN: %d samples, %d variables, %d edges",
        data.shape[0],
        data.shape[1],
        int(adjacency.sum().item()),
    )
    return data, adjacency


# Register with the real-world loader dispatch.
_register("syntren", load_syntren)
