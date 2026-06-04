import torch
from causal_meta.runners.metrics.graph import Metrics
from causal_meta.models.avici.model import AviciModel


def test_metrics_integration() -> None:
    targets = torch.randint(0, 2, (1, 5, 5)).float()
    samples = torch.randint(0, 2, (10, 1, 5, 5)).float()

    metrics = Metrics(metrics=["e-shd", "e-edgef1"])
    metrics.update(targets, samples)
    results = metrics.compute(summary_stats=False)

    assert "e-shd" in results
    assert "e-edgef1" in results
    assert isinstance(results["e-shd"], float)


def test_avici_loss():
    # Setup
    model = AviciModel(num_nodes=5, d_model=16, nhead=2, num_layers=2)

    # Batch=2, N=5
    logits = torch.randn(2, 5, 5)
    target = torch.randint(0, 2, (2, 5, 5)).float()

    loss = model.calculate_loss(logits, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0


def test_training_step_integration():
    # Setup
    model = AviciModel(num_nodes=5, d_model=16, nhead=2, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Fake batch
    batch_size = 2
    n_samples = 10
    n_nodes = 5
    x = torch.randn(batch_size, n_samples, n_nodes)
    adj = torch.randint(0, 2, (batch_size, n_nodes, n_nodes)).float()

    # Forward
    output = model(x)
    assert output.shape == (batch_size, n_nodes, n_nodes)

    # Loss
    loss = model.calculate_loss(output, adj)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert not torch.isnan(loss)
