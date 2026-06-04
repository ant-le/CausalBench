from __future__ import annotations

from typing import Iterator, Optional, Sized

import torch
from torch.utils.data import Sampler


class NoPaddingDistributedSampler(Sampler[int]):
    """
    Distributed sampler that shards indices across ranks **without padding**.

    This avoids the default `DistributedSampler` behavior of padding the dataset with
    duplicated indices when `len(dataset) % world_size != 0`, which can bias
    evaluation metrics.
    """

    def __init__(
        self,
        dataset: Sized,
        *,
        rank: Optional[int] = None,
        world_size: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

        if rank is None or world_size is None:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            else:
                rank = 0
                world_size = 1

        self.rank = int(rank)
        self.world_size = int(world_size)

        if self.world_size < 1:
            raise ValueError("world_size must be >= 1.")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(
                f"rank must be in [0, {self.world_size}). Got {self.rank}."
            )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        n = len(self.dataset)
        if n <= 0:
            return iter(())

        indices = list(range(n))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            perm = torch.randperm(n, generator=g).tolist()
            indices = [indices[i] for i in perm]

        return iter(indices[self.rank :: self.world_size])

    def __len__(self) -> int:
        n = len(self.dataset)
        if n <= self.rank:
            return 0
        return (n - self.rank + self.world_size - 1) // self.world_size
