import torch
import pytest

from causal_meta.datasets.utils import collate_fn_scm, normalize_scm_data


def test_normalize_scm_data_matches_collate_for_single_task() -> None:
    x = torch.tensor([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    expected = collate_fn_scm([{"seed": 0, "data": x, "adjacency": torch.zeros(2, 2)}])[
        "data"
    ].squeeze(0)
    out = normalize_scm_data(x)
    assert torch.allclose(out, expected, atol=1e-6)


def test_normalize_scm_data_standardizes_per_feature_for_batch() -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 8, 4) * 3.0 + 1.0
    out = normalize_scm_data(x)
    flat = out.reshape(-1, out.shape[-1])
    assert torch.allclose(flat.mean(dim=0), torch.zeros(4), atol=1e-6)
    assert torch.allclose(flat.std(dim=0, unbiased=False), torch.ones(4), atol=1e-6)


def test_normalize_scm_data_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError):
        normalize_scm_data(torch.zeros(1, 2, 3, 4))
