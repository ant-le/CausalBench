import torch

from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.generators.configs import FamilyConfig, DataModuleConfig
from causal_meta.datasets.generators.graphs import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms import LinearMechanismFactory
from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.torch_datasets import MetaFixedDataset, MetaIterableDataset
from causal_meta.datasets.utils import collate_fn_scm, compute_graph_hash


def _simple_family(n_nodes: int = 3) -> SCMFamily:
    generator = ErdosRenyiGenerator(edge_prob=0.4)
    mechanism = LinearMechanismFactory(weight_scale=0.1)
    return SCMFamily(
        name="test_simple",
        n_nodes=n_nodes,
        graph_generator=generator,
        mechanism_factory=mechanism,
    )


def test_erdos_renyi_generator_supports_sparsity_alias() -> None:
    generator = ErdosRenyiGenerator(sparsity=0.25)
    adjacency = generator(n_nodes=3, seed=0)
    assert adjacency.shape == (3, 3)
    assert generator.edge_prob == 0.25


def test_collate_fn_normalizes_batch() -> None:
    batch = [
        {
            "seed": 1,
            "data": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "adjacency": torch.zeros(2, 2),
        },
        {
            "seed": 2,
            "data": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "adjacency": torch.ones(2, 2),
        },
    ]
    out = collate_fn_scm(batch)
    normalized, adjs = out["data"], out["adjacency"]
    flat = normalized.reshape(-1, normalized.shape[-1])
    assert torch.allclose(flat.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(flat.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)
    assert adjs.shape == (2, 2, 2)


def test_meta_iterable_dataset_yields_tensors() -> None:
    family = _simple_family(n_nodes=4)
    dataset = MetaIterableDataset(family, base_seed=0, samples_per_task=5)
    item = next(iter(dataset))
    x, adj = item["data"], item["adjacency"]
    assert isinstance(x, torch.Tensor)
    assert isinstance(adj, torch.Tensor)
    assert x.shape == (5, 4)
    assert adj.shape == (4, 4)


def test_meta_fixed_dataset_is_deterministic() -> None:
    family = _simple_family(n_nodes=3)
    dataset = MetaFixedDataset(family, seeds=[42], samples_per_task=6)
    item1 = dataset[0]
    item2 = dataset[0]
    x1, adj1 = item1["data"], item1["adjacency"]
    x2, adj2 = item2["data"], item2["adjacency"]
    assert torch.allclose(x1, x2)
    assert torch.equal(adj1, adj2)


def test_causal_meta_module_initializes_and_sets_datasets() -> None:
    # Using explicit configs (The "Best Way")
    from causal_meta.datasets.generators.configs import (
        ErdosRenyiConfig,
        ScaleFreeConfig,
        LinearMechanismConfig,
        MLPMechanismConfig,
    )

    train_cfg = FamilyConfig(
        name="train",
        n_nodes=4,
        graph_cfg=ErdosRenyiConfig(sparsity=0.3),
        mech_cfg=LinearMechanismConfig(),
    )
    test_cfg = FamilyConfig(
        name="test",
        n_nodes=4,
        graph_cfg=ScaleFreeConfig(m=2),
        mech_cfg=MLPMechanismConfig(hidden_dim=8),
    )

    # Use the new explicit config structure
    config = DataModuleConfig(
        train_family=train_cfg,
        test_families={"test": test_cfg},
        seeds_val=[0, 1, 2],
        seeds_test=[10, 11, 12],
        base_seed=0,
        samples_per_task=4,
    )
    module = CausalMetaModule(config)
    module.setup()

    assert module.train_dataset is not None
    assert "test" in module.test_datasets
    assert module.test_datasets["test"] is not None
    assert "test" in module.family_distances
    entry = module.family_distances["test"]
    assert "spectral" in entry
    assert "kl_degree" in entry
    assert "mechanism" in entry


def test_causal_meta_module_dataloaders() -> None:
    # Passing raw dicts to test conversion (The "Compatibility Way")
    train_cfg = {
        "name": "train",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }
    test_cfg = {
        "name": "test",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }

    cfg = {
        "train_family": train_cfg,
        "test_families": {"test": test_cfg},
        "seeds_val": [0],
        "seeds_test": [1],
        "num_workers": 2,
        "pin_memory": True,
    }

    module = CausalMetaModule.from_config(cfg)
    module.setup()

    train_loader = module.train_dataloader()
    test_loaders = module.test_dataloader()

    assert train_loader.num_workers == 2
    assert train_loader.pin_memory is True

    assert "test" in test_loaders
    test_loader = test_loaders["test"]
    # We now expect test_loader to use configured workers, not 0
    assert test_loader.num_workers == 2
    assert test_loader.pin_memory is True
    assert test_loader.persistent_workers is False


def test_evaluation_dataloaders_are_cached_and_non_persistent() -> None:
    train_cfg = {
        "name": "train",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }
    test_cfg = {
        "name": "test",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }

    cfg = {
        "train_family": train_cfg,
        "test_families": {"test": test_cfg},
        "seeds_val": [0],
        "seeds_test": [1],
        "num_workers": 2,
        "pin_memory": True,
        "persistent_workers": True,
    }

    module = CausalMetaModule.from_config(cfg)
    module.setup()

    first_val_loaders = module.val_dataloader()
    second_val_loaders = module.val_dataloader()
    first_test_loaders = module.test_dataloader()
    second_test_loaders = module.test_dataloader()

    assert first_val_loaders is second_val_loaders
    assert first_test_loaders is second_test_loaders
    assert first_val_loaders["train"].persistent_workers is False
    assert first_test_loaders["test"].persistent_workers is False


def test_train_dataloader_iterates_with_collate_fn() -> None:
    train_cfg = {
        "name": "train",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }
    test_cfg = {
        "name": "test",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }

    cfg = {
        "train_family": train_cfg,
        "test_families": {"test": test_cfg},
        "seeds_val": [0],
        "seeds_test": [1],
        "num_workers": 0,
        "pin_memory": False,
        "samples_per_task": 8,
    }
    module = CausalMetaModule.from_config(cfg)
    train_loader = module.train_dataloader()

    batch = next(iter(train_loader))
    x, adj = batch["data"], batch["adjacency"]
    assert x.shape == (1, 8, 4)
    assert adj.shape == (1, 4, 4)
    assert batch["intervention_mask"].shape == (1, 8, 4)
    assert batch["node_mask"].shape == (1, 4)
    assert batch["sample_mask"].shape == (1, 8)

    flat = x.reshape(-1, x.shape[-1])
    assert torch.allclose(flat.mean(dim=0), torch.zeros(4), atol=1e-6)
    assert torch.allclose(flat.std(dim=0, unbiased=False), torch.ones(4), atol=1e-6)


def test_causal_meta_module_builds_validation_split_and_reserves_hashes() -> None:
    train_cfg = {
        "name": "train",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }
    test_cfg = {
        "name": "test",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }

    cfg = {
        "train_family": train_cfg,
        "test_families": {"test": test_cfg},
        "seeds_val": [2],
        "seeds_test": [1],
        "samples_per_task": 8,
        "num_workers": 0,
        "pin_memory": False,
    }

    module = CausalMetaModule.from_config(cfg)
    module.setup()

    val_loaders = module.val_dataloader()
    assert "train" in val_loaders

    assert module.train_dataset is not None
    val_hash = compute_graph_hash(
        module.val_families["train"].sample_task(2).adjacency_matrix
    )
    test_hash = compute_graph_hash(
        module.test_families["test"].sample_task(1).adjacency_matrix
    )
    assert val_hash in module.train_dataset.forbidden_hashes
    assert test_hash in module.train_dataset.forbidden_hashes


def test_collate_fn_scm_supports_padding_and_masks() -> None:
    batch = [
        {
            "seed": 1,
            "data": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "intervention_mask": torch.tensor([[0.0, 0.0], [0.0, 1.0]]),
            "adjacency": torch.zeros(2, 2),
        },
        {
            "seed": 2,
            "data": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "intervention_mask": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
            "adjacency": torch.ones(4, 4),
        },
    ]
    out = collate_fn_scm(batch, normalize=False)

    assert out["data"].shape == (2, 2, 4)
    assert out["intervention_mask"].shape == (2, 2, 4)
    assert out["adjacency"].shape == (2, 4, 4)
    assert out["node_mask"].shape == (2, 4)
    assert out["sample_mask"].shape == (2, 2)
    assert torch.equal(out["node_mask"][0], torch.tensor([False, False, True, True]))
    assert torch.equal(out["node_mask"][1], torch.tensor([False, False, False, False]))
    assert torch.equal(out["sample_mask"][0], torch.tensor([False, False]))
    assert torch.equal(out["sample_mask"][1], torch.tensor([False, True]))


def test_interventional_stream_produces_nonzero_intervention_mask() -> None:
    train_cfg = {
        "name": "train",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }
    test_cfg = {
        "name": "test",
        "n_nodes": 4,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }
    cfg = {
        "train_family": train_cfg,
        "test_families": {"test": test_cfg},
        "seeds_val": [0],
        "seeds_test": [1],
        "num_workers": 0,
        "pin_memory": False,
        "batch_size_train": 1,
        "samples_per_task_obs": 5,
        "samples_per_task_int": 3,
        "use_interventional_training": True,
        "train_p_obs_only": 0.0,
    }
    module = CausalMetaModule.from_config(cfg)
    batch = next(iter(module.train_dataloader()))
    assert batch["data"].shape == (1, 8, 4)
    assert float(batch["intervention_mask"].sum().item()) > 0.0


def test_train_n_nodes_bucket_batches() -> None:
    train_cfg = {
        "name": "train",
        "n_nodes": 3,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }
    test_cfg = {
        "name": "test",
        "n_nodes": 3,
        "graph_cfg": {"type": "er", "sparsity": 0.3},
        "mech_cfg": {"type": "linear"},
    }
    cfg = {
        "train_family": train_cfg,
        "test_families": {"test": test_cfg},
        "seeds_val": [0],
        "seeds_test": [1],
        "num_workers": 0,
        "pin_memory": False,
        "batch_size_train": 2,
        "train_n_nodes": [3, 5],
        "samples_per_task": 4,
    }
    module = CausalMetaModule.from_config(cfg)
    iterator = iter(module.train_dataloader())

    batch_1 = next(iterator)
    batch_2 = next(iterator)

    assert torch.unique(batch_1["n_nodes"]).numel() == 1
    assert torch.unique(batch_2["n_nodes"]).numel() == 1
    assert int(batch_1["n_nodes"][0].item()) != int(batch_2["n_nodes"][0].item())
