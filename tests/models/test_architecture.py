from unittest.mock import patch

import torch
import pytest
from causal_meta.models.base import BaseModel
from causal_meta.models.factory import ModelFactory, register_model

# Ensure models are registered by importing top-level package
import causal_meta.models


# Mock Model for testing
@register_model("mock_model")
class MockModel(BaseModel):
    def __init__(self, hidden_dim: int = 10, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer = torch.nn.Linear(1, 1)  # Dummy layer

    @property
    def needs_pretraining(self) -> bool:
        return True

    def forward(self, x: torch.Tensor, mask=None):
        return x

    def sample(self, x: torch.Tensor, num_samples: int = 1, mask=None):
        # Return dummy adjacency: (Batch, num_samples, V, V)
        # Assume x is (Batch, Samples, V)
        B, S, V = x.shape
        return torch.zeros(B, num_samples, V, V)

    def calculate_loss(
        self, output: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        # Dummy loss
        return torch.tensor([0.0], device=output.device).repeat(output.shape[0])


def test_base_model_interface():
    # Verify that MockModel instantiates and follows interface
    model = MockModel(hidden_dim=20)
    assert isinstance(model, BaseModel)
    assert model.hidden_dim == 20
    assert model.needs_pretraining is True


def test_factory_creation():
    config = {"type": "mock_model", "hidden_dim": 32}
    model = ModelFactory.create(config)
    assert isinstance(model, MockModel)
    assert model.hidden_dim == 32
    assert config["type"] == "mock_model"


def test_factory_invalid_type():
    config = {"type": "non_existent"}
    with pytest.raises(ValueError, match="Unknown model type"):
        ModelFactory.create(config)


def test_factory_missing_type():
    config = {"hidden_dim": 32}
    with pytest.raises(ValueError, match="must contain a 'type' key"):
        ModelFactory.create(config)


def test_avici_instantiation():
    config = {
        "type": "avici",
        "num_nodes": 10,
        "d_model": 16,
        "nhead": 2,
        "num_layers": 2,
    }
    model = ModelFactory.create(config)
    assert model.d_model == 16
    assert isinstance(model, causal_meta.models.AviciModel)
    assert model.needs_pretraining is True


def test_bcnp_instantiation():
    config = {
        "type": "bcnp",
        "num_nodes": 10,
        "d_model": 32,
        "nhead": 4,
        "num_layers": 2,
    }
    model = ModelFactory.create(config)
    assert model.d_model == 32
    assert isinstance(model, causal_meta.models.BCNP)
    assert model.needs_pretraining is True


def test_bcnp_forward_handles_non_contiguous_projection_layout():
    model = causal_meta.models.BCNP(
        num_nodes=5,
        d_model=32,
        nhead=4,
        num_layers=2,
        num_layers_decoder=2,
    )
    x = torch.randn(2, 3, 5)
    output = model(x)

    assert output.shape == (model.n_perm_samples, 2, 5, 5)


def test_avici_sample_has_zero_diagonal():
    model = causal_meta.models.AviciModel(num_nodes=5, d_model=8, nhead=2, num_layers=2)
    x = torch.randn(3, 7, 5)
    samples = model.sample(x, num_samples=4)
    assert samples.shape == (3, 4, 5, 5)
    diag = torch.diagonal(samples, dim1=-2, dim2=-1)
    assert torch.all(diag == 0)


def test_avici_loss_calculation():
    model = causal_meta.models.AviciModel(num_nodes=5, d_model=8, nhead=2, num_layers=2)
    # Mock data
    logits = torch.randn(3, 5, 5)  # (Batch, N, N)
    target = torch.randint(0, 2, (3, 5, 5)).float()

    # Should fail if cyclicity is missing or logic is wrong
    loss = model.calculate_loss(logits, target, update_regulariser=True)
    assert loss.dim() == 0


def test_avici_acyclicity_regularizer_is_differentiable() -> None:
    model = causal_meta.models.AviciModel(num_nodes=5, d_model=8, nhead=2, num_layers=2)
    model.regulariser_weight.data = torch.tensor(1.0)

    logits = (0.1 * torch.randn(2, 5, 5)).requires_grad_(True)
    target = torch.sigmoid(logits.detach())

    loss = model.calculate_loss(logits, target, update_regulariser=False)
    loss.backward()

    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_avici_backward_safe_with_regulariser_update() -> None:
    """Verify that backward() succeeds when update_regulariser=True.

    Regression test: inplace updates to ``regulariser_weight`` inside
    ``calculate_loss`` previously caused a RuntimeError during backward
    because the autograd graph version check failed.
    """
    torch.manual_seed(0)
    model = causal_meta.models.AviciModel(
        num_nodes=5,
        d_model=8,
        nhead=2,
        num_layers=2,
        regulariser_lr=1e-4,
        regulariser_ema_alpha=1.0,
        regulariser_warmup_updates=0,
    )
    model.regulariser_weight.data = torch.tensor(1.0)

    x = torch.randn(2, 50, 5)
    target = torch.zeros(2, 5, 5)

    output = model(x)
    loss = model.calculate_loss(output, target, update_regulariser=True).mean()
    loss.backward()  # should not raise RuntimeError


def test_avici_regulariser_update_uses_warmup() -> None:
    model = causal_meta.models.AviciModel(
        num_nodes=5,
        d_model=8,
        nhead=2,
        num_layers=2,
        regulariser_lr=1.0,
        regulariser_ema_alpha=1.0,
        regulariser_warmup_updates=4,
    )

    # Drive the weight upward with a known positive penalty so the test does
    # not depend on the random seed used inside the spectral power iteration.
    penalty = torch.tensor(2.0)

    model.update_regulariser_weight(penalty)
    first_weight = float(model.regulariser_weight.item())
    model.update_regulariser_weight(penalty)
    second_weight = float(model.regulariser_weight.item())

    assert first_weight > 0.0
    assert second_weight > first_weight
    assert int(model.regulariser_update_count.item()) == 2


@patch("torch.distributed.is_available")
@patch("torch.distributed.is_initialized")
@patch("torch.distributed.get_world_size")
@patch("torch.distributed.all_reduce")
def test_avici_regulariser_update_reduces_across_ranks(
    mock_all_reduce, mock_world_size, mock_init, mock_avail
) -> None:
    mock_avail.return_value = True
    mock_init.return_value = True
    mock_world_size.return_value = 2

    def side_effect(tensor: torch.Tensor, op=None) -> None:
        _ = op
        tensor.fill_(6.0)

    mock_all_reduce.side_effect = side_effect

    model = causal_meta.models.AviciModel(
        num_nodes=5,
        d_model=8,
        nhead=2,
        num_layers=2,
        regulariser_lr=1.0,
        regulariser_ema_alpha=1.0,
        regulariser_warmup_updates=1,
    )

    model.update_regulariser_weight(torch.tensor(1.0))

    assert float(model.regulariser_weight.item()) == pytest.approx(3.0)
