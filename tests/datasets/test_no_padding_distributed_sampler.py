from causal_meta.datasets.utils.sampling import NoPaddingDistributedSampler


def test_no_padding_distributed_sampler_shards_without_duplicates() -> None:
    dataset = list(range(10))
    world_size = 3

    all_indices: list[int] = []
    for rank in range(world_size):
        sampler = NoPaddingDistributedSampler(
            dataset, rank=rank, world_size=world_size, shuffle=False
        )
        indices = list(iter(sampler))
        assert len(indices) == len(sampler)
        all_indices.extend(indices)

    assert sorted(all_indices) == list(range(10))
    assert len(all_indices) == len(set(all_indices))


def test_no_padding_distributed_sampler_len_matches_indices() -> None:
    dataset = list(range(5))
    sampler = NoPaddingDistributedSampler(dataset, rank=2, world_size=4, shuffle=False)
    assert list(iter(sampler)) == [2]
    assert len(sampler) == 1

