from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict, Mapping

from causal_meta.datasets.generators import configs

# Optional Hydra support
try:
    from hydra.utils import instantiate as hydra_instantiate
    from omegaconf import DictConfig, OmegaConf
except ImportError:
    hydra_instantiate = None
    DictConfig = None
    OmegaConf = None


class _HydraConfigWrapper:
    def __init__(self, cfg: Mapping[str, Any]) -> None:
        self.cfg = cfg

    def instantiate(self) -> Any:
        if hydra_instantiate is None:
            raise RuntimeError("Hydra is not installed but '_target_' was found.")
        return hydra_instantiate(self.cfg, _recursive_=True)


class _DirectObjectWrapper:
    def __init__(self, obj: Any) -> None:
        self.obj = obj

    def instantiate(self) -> Any:
        return self.obj


def _coerce_dict(cfg: Any) -> Any:
    if DictConfig is not None and isinstance(cfg, DictConfig):
        from omegaconf import OmegaConf as _OmegaConf

        return _OmegaConf.to_container(cfg, resolve=True)
    return cfg


def _exclude_type(d: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k != "type"}


def _dataclass_field_names(config_cls: type[Any]) -> set[str]:
    if not is_dataclass(config_cls):
        return set()
    return {field.name for field in fields(config_cls)}


def _typed_config_kwargs(
    cfg: Mapping[str, Any],
    config_cls: type[Any],
    *,
    known_type_keys: set[str],
    kind: str,
) -> Dict[str, Any]:
    """Return constructor kwargs, dropping stale keys from Hydra dict merges.

    Hydra deep-merges nested dictionaries when a config inherits another config.
    If an override changes only ``type`` (for example ``sbm`` to ``er``), keys
    from the previous type can remain in the composed mapping.  Drop keys that
    are valid for another generator type, but keep failing on truly unknown keys.
    """

    kwargs = _exclude_type(cfg)
    field_names = _dataclass_field_names(config_cls)
    if not field_names:
        return kwargs

    unexpected_keys = set(kwargs) - field_names
    invalid_keys = unexpected_keys - known_type_keys
    if invalid_keys:
        accepted = ", ".join(sorted(field_names))
        invalid = ", ".join(sorted(invalid_keys))
        raise TypeError(
            f"Unexpected {kind} config key(s) for type '{cfg.get('type')}': "
            f"{invalid}. Accepted keys: {accepted}."
        )

    return {key: value for key, value in kwargs.items() if key in field_names}


GRAPH_CONFIG_MAP = {
    "er": configs.ErdosRenyiConfig,
    "sf": configs.ScaleFreeConfig,
    "scale_free": configs.ScaleFreeConfig,
    "sbm": configs.SBMConfig,
    "ws": configs.WattsStrogatzConfig,
    "watts_strogatz": configs.WattsStrogatzConfig,
    "grg": configs.GeometricRandomConfig,
    "geometric_random": configs.GeometricRandomConfig,
    "mixture": configs.MixtureGraphConfig,
}

GRAPH_CONFIG_KEYS = {
    key
    for config_cls in set(GRAPH_CONFIG_MAP.values())
    for key in _dataclass_field_names(config_cls)
}

MECHANISM_CONFIG_MAP = {
    "linear": configs.LinearMechanismConfig,
    "mlp": configs.MLPMechanismConfig,
    "mixture": configs.MixtureMechanismConfig,
    "square": configs.SquareMechanismConfig,
    "periodic": configs.PeriodicMechanismConfig,
    "logistic": configs.LogisticMapMechanismConfig,
    "logistic_map": configs.LogisticMapMechanismConfig,
    "gp": configs.GPMechanismConfig,
    "pnl": configs.PNLMechanismConfig,
}

MECHANISM_CONFIG_KEYS = {
    key
    for config_cls in set(MECHANISM_CONFIG_MAP.values())
    for key in _dataclass_field_names(config_cls)
}


def load_graph_config(cfg: Any) -> configs.GraphConfig:
    cfg = _coerce_dict(cfg)

    if hasattr(cfg, "instantiate"):
        return cfg  # type: ignore

    if callable(cfg) and not isinstance(cfg, Mapping):
        return _DirectObjectWrapper(cfg)

    if not isinstance(cfg, Mapping):
        raise TypeError(
            f"Graph config must be a mapping, callable, or config object. Got {type(cfg)}"
        )

    if "_target_" in cfg:
        return _HydraConfigWrapper(cfg)

    type_name = cfg.get("type")
    if not type_name:
        raise ValueError("Graph config must contain a 'type' or '_target_' key.")

    if type_name == "mixture":
        kwargs = _typed_config_kwargs(
            cfg,
            configs.MixtureGraphConfig,
            known_type_keys=GRAPH_CONFIG_KEYS,
            kind="graph",
        )
        if "generators" not in kwargs:
            raise ValueError("Mixture graph config must contain 'generators' list.")
        kwargs["generators"] = [load_graph_config(g) for g in kwargs["generators"]]
        return configs.MixtureGraphConfig(**kwargs)

    config_cls = GRAPH_CONFIG_MAP.get(str(type_name))
    if config_cls is None:
        raise ValueError(f"Unknown graph generator type: '{type_name}'")

    return config_cls(
        **_typed_config_kwargs(
            cfg,
            config_cls,
            known_type_keys=GRAPH_CONFIG_KEYS,
            kind="graph",
        )
    )


def load_mechanism_config(cfg: Any) -> configs.MechanismConfig:
    cfg = _coerce_dict(cfg)

    if hasattr(cfg, "instantiate"):
        return cfg  # type: ignore

    if callable(cfg) and not isinstance(cfg, Mapping):
        return _DirectObjectWrapper(cfg)

    if not isinstance(cfg, Mapping):
        raise TypeError(
            f"Mechanism config must be a mapping, callable, or config object. Got {type(cfg)}"
        )

    if "_target_" in cfg:
        return _HydraConfigWrapper(cfg)

    type_name = cfg.get("type")
    if not type_name:
        raise ValueError("Mechanism config must contain a 'type' or '_target_' key.")

    if type_name == "mixture":
        kwargs = _typed_config_kwargs(
            cfg,
            configs.MixtureMechanismConfig,
            known_type_keys=MECHANISM_CONFIG_KEYS,
            kind="mechanism",
        )
        if "factories" not in kwargs:
            raise ValueError("Mixture mechanism config must contain 'factories' list.")
        kwargs["factories"] = [load_mechanism_config(f) for f in kwargs["factories"]]
        return configs.MixtureMechanismConfig(**kwargs)

    # PNL supports a nested inner mechanism config.
    # Coerce it recursively so downstream code can call `.instantiate()`.
    if type_name in {"pnl"}:
        kwargs = _typed_config_kwargs(
            cfg,
            configs.PNLMechanismConfig,
            known_type_keys=MECHANISM_CONFIG_KEYS,
            kind="mechanism",
        )
        inner = kwargs.get("inner_config")
        if inner is not None:
            kwargs["inner_config"] = load_mechanism_config(inner)
        return configs.PNLMechanismConfig(**kwargs)

    config_cls = MECHANISM_CONFIG_MAP.get(str(type_name))
    if config_cls is None:
        raise ValueError(f"Unknown mechanism factory type: '{type_name}'")

    return config_cls(
        **_typed_config_kwargs(
            cfg,
            config_cls,
            known_type_keys=MECHANISM_CONFIG_KEYS,
            kind="mechanism",
        )
    )


def load_family_config(
    cfg: Any,
    *,
    default_n_nodes: int | None = None,
    expected_name: str | None = None,
) -> configs.FamilyConfig | configs.RealWorldFamilyConfig:
    cfg = _coerce_dict(cfg)

    if isinstance(cfg, (configs.FamilyConfig, configs.RealWorldFamilyConfig)):
        return cfg

    if not isinstance(cfg, Mapping):
        raise TypeError("Family config must be a dict or FamilyConfig object.")

    # ---------- real-world branch ----------
    family_type = cfg.get("type")
    if family_type == "real_world":
        loader = cfg.get("loader")
        if not loader:
            raise ValueError("Real-world family config must provide a 'loader' key.")
        name = str(cfg.get("name", "")).strip()
        if not name:
            raise ValueError("Family config must provide a non-empty 'name'.")
        if expected_name is not None and name != expected_name:
            raise ValueError(
                f"Family config name mismatch: expected '{expected_name}', got '{name}'."
            )
        n_nodes_raw = cfg.get("n_nodes", default_n_nodes)
        if n_nodes_raw is None:
            raise ValueError(
                "Real-world family config must provide 'n_nodes' or a top-level "
                "data.n_nodes must be set."
            )
        samples_per_task_raw = cfg.get("samples_per_task")
        loader_kwargs_raw = cfg.get("loader_kwargs")
        inference_n_samples_raw = cfg.get("inference_n_samples")
        rw_cfg = configs.RealWorldFamilyConfig(
            name=name,
            loader=str(loader),
            n_nodes=int(n_nodes_raw),
            samples_per_task=(
                int(samples_per_task_raw) if samples_per_task_raw is not None else None
            ),
            loader_kwargs=(
                dict(loader_kwargs_raw) if loader_kwargs_raw is not None else None
            ),
            inference_n_samples=(
                int(inference_n_samples_raw)
                if inference_n_samples_raw is not None
                else None
            ),
        )
        rw_cfg.validate()
        return rw_cfg

    # ---------- generative SCM branch (existing logic) ----------
    graph_cfg = cfg.get("graph_cfg", cfg.get("graph"))
    mech_cfg = cfg.get("mech_cfg", cfg.get("mech"))

    if graph_cfg is None or mech_cfg is None:
        raise ValueError("Family config must provide 'graph_cfg' and 'mech_cfg'.")

    n_nodes_raw = cfg.get("n_nodes", default_n_nodes)
    if n_nodes_raw is None:
        raise ValueError(
            "Family config must provide 'n_nodes' or a top-level data.n_nodes must be set."
        )

    name = str(cfg.get("name", "")).strip()
    if not name:
        raise ValueError("Family config must provide a non-empty 'name'.")
    if expected_name is not None and name != expected_name:
        raise ValueError(
            f"Family config name mismatch: expected '{expected_name}', got '{name}'."
        )

    samples_per_task_raw = cfg.get("samples_per_task")
    noise_type = str(cfg.get("noise_type", "gaussian")).strip()
    inference_n_samples_raw = cfg.get("inference_n_samples")
    family_cfg = configs.FamilyConfig(
        name=name,
        n_nodes=int(n_nodes_raw),
        graph_cfg=load_graph_config(graph_cfg),
        mech_cfg=load_mechanism_config(mech_cfg),
        samples_per_task=(
            int(samples_per_task_raw) if samples_per_task_raw is not None else None
        ),
        noise_type=noise_type,
        inference_n_samples=(
            int(inference_n_samples_raw)
            if inference_n_samples_raw is not None
            else None
        ),
    )
    family_cfg.validate()
    return family_cfg


def load_data_module_config(cfg: Any) -> configs.DataModuleConfig:
    cfg = _coerce_dict(cfg)

    if isinstance(cfg, configs.DataModuleConfig):
        return cfg

    if not isinstance(cfg, Mapping):
        raise TypeError("DataModule config must be a dict or DataModuleConfig object.")

    default_n_nodes = cfg.get("n_nodes", cfg.get("num_nodes", None))
    train_family = load_family_config(
        cfg["train_family"], default_n_nodes=default_n_nodes
    )

    test_families_raw = cfg.get("test_families")
    if test_families_raw is None:
        raise ValueError("Config must contain 'test_families'.")

    if not isinstance(test_families_raw, Mapping):
        raise TypeError("'test_families' must be a dictionary of configs.")

    test_families: Dict[str, configs.AnyFamilyConfig] = {}
    for name, sub_cfg in test_families_raw.items():
        if str(name).startswith("_"):
            continue
        family_cfg = load_family_config(
            sub_cfg,
            default_n_nodes=default_n_nodes,
            expected_name=str(name),
        )
        if family_cfg.name in test_families:
            raise ValueError(f"Duplicate test family name: '{family_cfg.name}'.")
        test_families[family_cfg.name] = family_cfg

    val_families_raw = cfg.get("val_families")
    if val_families_raw is None:
        val_families: Dict[str, configs.FamilyConfig] = {}
    else:
        if not isinstance(val_families_raw, Mapping):
            raise TypeError("'val_families' must be a dictionary of configs.")
        val_families = {}
        for name, sub_cfg in val_families_raw.items():
            if str(name).startswith("_"):
                continue
            family_cfg = load_family_config(
                sub_cfg,
                default_n_nodes=default_n_nodes,
                expected_name=str(name),
            )
            if family_cfg.name in val_families:
                raise ValueError(
                    f"Duplicate validation family name: '{family_cfg.name}'."
                )
            val_families[family_cfg.name] = family_cfg

    allowed_keys = {
        "seeds_test",
        "seeds_val",
        "base_seed",
        "samples_per_task",
        "inference_n_samples",
        "samples_per_task_obs",
        "samples_per_task_int",
        "use_interventional_training",
        "train_p_obs_only",
        "intervention_value",
        "train_n_nodes",
        "safety_checks",
        "num_workers",
        "pin_memory",
        "persistent_workers",
        "prefetch_factor",
        "normalize_data",
        "batch_size_train",
        "batch_size_val",
        "batch_size_test",
        "batch_size_test_interventional",
        "hash_mechanisms",
    }

    kwargs = {k: v for k, v in cfg.items() if k in allowed_keys}

    if "seeds_test" not in kwargs or "seeds_val" not in kwargs:
        raise ValueError("Config must contain 'seeds_test' and 'seeds_val'.")

    return configs.DataModuleConfig(
        train_family=train_family,
        val_families=val_families,
        test_families=test_families,
        **kwargs,
    )
