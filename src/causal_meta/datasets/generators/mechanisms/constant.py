from __future__ import annotations

import torch
from torch import nn


class ConstantMechanism(nn.Module):
    """Mechanism that outputs a fixed value, ignoring parents and noise."""

    def __init__(self, value: float | torch.Tensor) -> None:
        super().__init__()
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        self.register_buffer("value", value)

    def forward(self, parents: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # parents: (batch_size, n_parents)
        # noise: (batch_size, 1)
        batch_size = noise.shape[0]
        
        # Broadcast the scalar/single-element tensor to the batch size
        if self.value.numel() == 1:
            return self.value.expand(batch_size)
            
        # If the value is a vector, we assume it's already (batch_size,) or compatible
        # This allows for 'contextual' interventions if needed, though rare in simple 'do'.
        if self.value.shape[0] == batch_size:
             return self.value
             
        # Fallback/Broadcast if dimensions allow
        return self.value.expand(batch_size)
