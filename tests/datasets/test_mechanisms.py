import torch

from causal_meta.datasets.generators.mechanisms import MLPMechanism


def test_mlp_mechanism_uses_noise_and_parents() -> None:
    torch.manual_seed(0)
    mechanism = MLPMechanism(input_dim=3, hidden_dim=5)

    parents = torch.randn(4, 3)
    noise = torch.randn(4, 1)

    output = mechanism(parents, noise)
    expected = mechanism.net(torch.cat([parents, noise], dim=1)).squeeze(-1)

    assert output.shape == (4,)
    assert torch.allclose(output, expected)


def test_mlp_mechanism_handles_no_parents() -> None:
    torch.manual_seed(1)
    mechanism = MLPMechanism(input_dim=0, hidden_dim=4)

    parents = torch.zeros(3, 0)
    noise = torch.randn(3, 1)

    output = mechanism(parents, noise)
    expected = mechanism.net(torch.cat([parents, noise], dim=1)).squeeze(-1)

    assert output.shape == (3,)
    assert torch.allclose(output, expected)
