from causal_meta.datasets.data_module import CausalMetaModule
from causal_meta.datasets.generators.graphs import ErdosRenyiGenerator
from causal_meta.datasets.generators.mechanisms import LinearMechanismFactory
from causal_meta.datasets.scm import SCMFamily
from causal_meta.datasets.utils import compute_graph_hash


class CountingMechanismFactory(LinearMechanismFactory):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return super().__call__(*args, **kwargs)


def test_disjointness_hashing_does_not_build_mechanisms() -> None:
    dummy_cfg = {
        "train_family": {
            "name": "train",
            "n_nodes": 3,
            "graph_cfg": {"type": "er", "sparsity": 0.3},
            "mech_cfg": {"type": "linear"},
        },
        "test_families": {
            "test": {
                "name": "test",
                "n_nodes": 3,
                "graph_cfg": {"type": "er", "sparsity": 0.3},
                "mech_cfg": {"type": "linear"},
            }
        },
        "seeds_val": [0],
        "seeds_test": [1],
    }
    module = CausalMetaModule.from_config(dummy_cfg)

    mech = CountingMechanismFactory(weight_scale=0.1)
    family = SCMFamily(
        name="test_counting",
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.4),
        mechanism_factory=mech,
    )
    _ = module._sample_hashes(family, seeds=[0, 1, 2])
    assert mech.calls == 0


def test_mechanism_aware_hashing_produces_different_hashes() -> None:
    """Test that same DAG with different mechanisms produces different hashes."""
    family = SCMFamily(
        name="test_mech_hash",
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.5),
        mechanism_factory=LinearMechanismFactory(weight_scale=1.0),
    )

    # Sample the same graph structure twice (same seed)
    instance1 = family.sample_task(seed=42)
    instance2 = family.sample_task(seed=42)

    # Structure-only hashes should be identical
    hash1_struct = compute_graph_hash(instance1.adjacency_matrix)
    hash2_struct = compute_graph_hash(instance2.adjacency_matrix)
    assert hash1_struct == hash2_struct

    # With mechanism hashing, they should also be identical (same seed = same mechanisms)
    hash1_mech = compute_graph_hash(
        instance1.adjacency_matrix,
        mechanisms=instance1.mechanisms,
        include_mechanisms=True,
    )
    hash2_mech = compute_graph_hash(
        instance2.adjacency_matrix,
        mechanisms=instance2.mechanisms,
        include_mechanisms=True,
    )
    assert hash1_mech == hash2_mech

    # Now sample with different seed - same structure possible, different mechanisms
    # Use a family with fixed structure but different mechanism weights
    instance3 = family.sample_task(seed=43)

    # If structures happen to be the same, mechanism hashes should differ
    hash3_struct = compute_graph_hash(instance3.adjacency_matrix)
    hash3_mech = compute_graph_hash(
        instance3.adjacency_matrix,
        mechanisms=instance3.mechanisms,
        include_mechanisms=True,
    )

    # Different seeds produce different mechanism parameters,
    # so mechanism-aware hash should differ from hash1_mech
    # (assuming structures are different, which is highly likely with different seeds)
    if hash1_struct == hash3_struct:
        # Same structure but different mechanisms -> different hash
        assert hash1_mech != hash3_mech


def test_hash_mechanisms_config_enables_mechanism_hashing() -> None:
    """Test that hash_mechanisms config option affects _sample_hashes behavior."""
    # Create a counting factory to track mechanism builds
    # Note: __call__ is invoked once per sample_task (which builds all node mechanisms)
    mech_factory = CountingMechanismFactory(weight_scale=0.1)
    family = SCMFamily(
        name="test_hash_config",
        n_nodes=3,
        graph_generator=ErdosRenyiGenerator(edge_prob=0.4),
        mechanism_factory=mech_factory,
    )

    cfg_base = {
        "train_family": {
            "name": "train",
            "n_nodes": 3,
            "graph_cfg": {"type": "er", "sparsity": 0.5},
            "mech_cfg": {"type": "linear"},
        },
        "test_families": {
            "test": {
                "name": "test",
                "n_nodes": 3,
                "graph_cfg": {"type": "er", "sparsity": 0.5},
                "mech_cfg": {"type": "linear"},
            }
        },
        "seeds_val": [0],
        "seeds_test": [1, 2, 3],
    }

    # Without mechanism hashing - no mechanism calls on external family
    cfg_no_mech = {**cfg_base, "hash_mechanisms": False}
    module_no_mech = CausalMetaModule.from_config(cfg_no_mech)

    mech_factory.calls = 0
    _ = module_no_mech._sample_hashes(family, seeds=[10, 11, 12])
    assert mech_factory.calls == 0, "hash_mechanisms=False should not build mechanisms"

    # With mechanism hashing - mechanism factory __call__ should be invoked
    cfg_with_mech = {**cfg_base, "hash_mechanisms": True}
    module_with_mech = CausalMetaModule.from_config(cfg_with_mech)

    mech_factory.calls = 0
    _ = module_with_mech._sample_hashes(family, seeds=[10, 11, 12])
    # __call__ is invoked once per sample_task (i.e., once per seed)
    assert mech_factory.calls == 3, (
        f"hash_mechanisms=True should build mechanisms for each seed, got {mech_factory.calls}"
    )
