import torch

from causal_meta.runners.utils.distributed import select_device


def test_select_device_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device = select_device(local_rank=1, is_distributed=False)
    assert device.type == "cuda"
    assert device.index == 1


def test_select_device_uses_mps_when_single_process(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device = select_device(local_rank=0, is_distributed=False)
    assert device.type == "mps"


def test_select_device_uses_cpu_when_distributed_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    device = select_device(local_rank=0, is_distributed=True)
    assert device.type == "cpu"


def test_select_device_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device = select_device(local_rank=0, is_distributed=False)
    assert device.type == "cpu"
