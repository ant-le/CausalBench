"""Sachs et al. (2005) protein signalling network.

11 variables (phosphoproteins and phospholipids), 17 directed edges in the
consensus DAG.  Observational data (~853 samples) is downloaded from the
bnlearn data repository on first use and cached locally.

Reference:
    Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P.
    (2005).  Causal protein-signaling networks derived from multiparameter
    single-cell data.  *Science*, 308(5721), 523-529.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import urllib.request
from pathlib import Path
from typing import Tuple

import torch

from causal_meta.datasets.real_world.registry import _register

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Consensus DAG  (Sachs et al., 2005 — 17 directed edges)
# ---------------------------------------------------------------------------

# Variable ordering used in the bnlearn Sachs dataset.
SACHS_VARIABLES = [
    "Raf",  # 0
    "Mek",  # 1
    "Plcg",  # 2
    "PIP2",  # 3
    "PIP3",  # 4
    "Erk",  # 5
    "Akt",  # 6
    "PKA",  # 7
    "PKC",  # 8
    "P38",  # 9
    "Jnk",  # 10
]

# Edges as (source_idx, target_idx).  The consensus DAG from Sachs et al.
# These edges are the widely accepted ground-truth network used in benchmarks
# by DiBS, BayesDAG, and others.
SACHS_EDGES: list[tuple[int, int]] = [
    (2, 4),  # Plcg  -> PIP3
    (2, 3),  # Plcg  -> PIP2
    (4, 3),  # PIP3  -> PIP2
    (8, 1),  # PKC   -> Mek
    (8, 0),  # PKC   -> Raf
    (8, 7),  # PKC   -> PKA
    (8, 9),  # PKC   -> P38
    (8, 10),  # PKC   -> Jnk
    (7, 0),  # PKA   -> Raf
    (7, 1),  # PKA   -> Mek
    (7, 5),  # PKA   -> Erk
    (7, 6),  # PKA   -> Akt
    (7, 9),  # PKA   -> P38
    (7, 10),  # PKA   -> Jnk
    (0, 1),  # Raf   -> Mek
    (1, 5),  # Mek   -> Erk
    (5, 6),  # Erk   -> Akt
]

_SACHS_N_NODES = 11
_SACHS_N_EDGES = 17

# bnlearn data URL (observational-only, tab-separated, gzip-compressed).
_SACHS_DATA_URL = "https://www.bnlearn.com/book-crc/code/sachs.data.txt.gz"

# Local cache directory (under package data).
_CACHE_DIR = Path(__file__).resolve().parent / "_cache"


def _sachs_adjacency() -> torch.Tensor:
    """Build the 11×11 consensus adjacency matrix."""
    adj = torch.zeros(_SACHS_N_NODES, _SACHS_N_NODES, dtype=torch.float32)
    for src, tgt in SACHS_EDGES:
        adj[src, tgt] = 1.0
    assert int(adj.sum().item()) == _SACHS_N_EDGES
    return adj


def _download_sachs_data(cache_path: Path) -> torch.Tensor:
    """Download Sachs observational data from bnlearn and cache as CSV."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Downloading Sachs dataset from %s ...", _SACHS_DATA_URL)
    with urllib.request.urlopen(_SACHS_DATA_URL, timeout=30) as resp:
        compressed = resp.read()

    # The upstream file is gzip-compressed.
    raw = gzip.decompress(compressed).decode("utf-8")

    # Parse the tab-separated data.
    reader = csv.reader(io.StringIO(raw), delimiter="\t")
    header = next(reader)

    # Build column mapping to ensure correct variable ordering.
    col_map = {name.strip(): idx for idx, name in enumerate(header)}
    col_order = [col_map[v] for v in SACHS_VARIABLES]

    rows: list[list[float]] = []
    for row in reader:
        if not row:
            continue
        values = [float(row[c]) for c in col_order]
        rows.append(values)

    data = torch.tensor(rows, dtype=torch.float32)

    # Cache the reordered data as a simple text file.
    with open(cache_path, "w") as f:
        for row_vals in rows:
            f.write("\t".join(f"{v:.6f}" for v in row_vals) + "\n")
    log.info(
        "Sachs dataset cached to %s  (%d samples, %d variables)",
        cache_path,
        data.shape[0],
        data.shape[1],
    )
    return data


def _load_cached_data(cache_path: Path) -> torch.Tensor:
    """Load previously cached Sachs data."""
    rows: list[list[float]] = []
    with open(cache_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(v) for v in line.split("\t")])
    return torch.tensor(rows, dtype=torch.float32)


def load_sachs(**kwargs: object) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load the Sachs (2005) protein signalling dataset.

    Returns:
        ``(data, adjacency)`` — data is ``(N, 11)`` float32, adjacency is
        ``(11, 11)`` float32.
    """
    cache_path = _CACHE_DIR / "sachs_obs.tsv"
    if cache_path.exists():
        data = _load_cached_data(cache_path)
    else:
        data = _download_sachs_data(cache_path)

    adjacency = _sachs_adjacency()

    assert data.shape[1] == _SACHS_N_NODES, (
        f"Expected {_SACHS_N_NODES} columns, got {data.shape[1]}"
    )
    return data, adjacency


# Register with the real-world loader dispatch.
_register("sachs", load_sachs)
