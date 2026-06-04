from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf


def _escape_tex(value: str) -> str:
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        value = value.replace(char, repl)
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at {path}")
    return data


def _fmt(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_fmt(v) for v in value)
    if isinstance(value, dict):
        parts = [f"{k}={_fmt(v)}" for k, v in value.items()]
        return "; ".join(parts)
    return str(value)


def _graph_summary(cfg: Mapping[str, Any]) -> str:
    graph_type = str(cfg.get("type", ""))
    if graph_type == "er":
        return f"ER ({cfg.get('sparsity')})"
    if graph_type == "sf":
        return f"SF (m={cfg.get('m')})"
    if graph_type == "sbm":
        return f"SBM (b={cfg.get('n_blocks')}, in={cfg.get('p_intra')}, out={cfg.get('p_inter')})"
    if graph_type == "mixture":
        generators = cfg.get("generators", [])
        return f"Mixture ({len(generators)} generators)"
    return _fmt(cfg)


def _mech_summary(cfg: Mapping[str, Any]) -> str:
    mech_type = str(cfg.get("type", ""))
    pretty = mech_type.replace("_", " ").title()
    if mech_type == "mlp":
        return f"MLP (h={cfg.get('hidden_dim')})"
    if mech_type == "gp":
        return f"RFF ({cfg.get('mode')}, rff={cfg.get('rff_dim')})"
    if mech_type == "pnl":
        return "PNL"
    if mech_type == "mixture":
        factories = cfg.get("factories", [])
        return f"Mixture ({len(factories)} factories)"
    return pretty if mech_type else _fmt(cfg)


def _write_table(
    output_path: Path,
    *,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[str]],
    colspec: str,
    resize: bool = False,
    font_size: str = r"\footnotesize",
) -> None:
    lines: list[str] = []
    if font_size:
        lines.append(font_size)
    if resize:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{colspec}}}")
    lines.append(r"\toprule")
    header_line = " & ".join(rf"\textbf{{{_escape_tex(h)}}}" for h in headers) + r" \\"
    lines.append(header_line)
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(_escape_tex(cell) for cell in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if resize:
        lines.append(r"}")
    output_path.write_text("\n".join(lines) + "\n")


def _write_family_table(
    output_path: Path,
    *,
    caption: str,
    label: str,
    rows: list[list[str]],
) -> None:
    _write_table(
        output_path,
        caption=caption,
        label=label,
        headers=["Family", "Category", "Graph", "Mechanism", "N_G", "N"],
        rows=rows,
        colspec=(
            ">{\\raggedright\\arraybackslash}p{3.2cm}"
            ">{\\raggedright\\arraybackslash}p{1.8cm}"
            ">{\\raggedright\\arraybackslash}p{2.5cm}"
            ">{\\raggedright\\arraybackslash}p{2.3cm}cc"
        ),
        resize=True,
        font_size=r"\scriptsize",
    )


def _family_category(name: str) -> str:
    if name.startswith("id_"):
        return "ID"
    if name.startswith("ood_graph_"):
        return "Graph-OOD"
    if name.startswith("ood_mech_"):
        return "Mechanism-OOD"
    if name.startswith("ood_noise_"):
        return "Noise-OOD"
    if name.startswith("ood_both_"):
        return "Compound-OOD"
    if name.startswith("ood_nodes_"):
        return "Node-count"
    if name.startswith("ood_samples_"):
        return "Sample-count"
    return "Other"


def _family_label(name: str) -> str:
    if name.startswith("id_"):
        if "linear" in name and "er20" in name:
            return "ID: Linear ER20"
        if "neuralnet" in name and "sf2" in name:
            return "ID: MLP SF2"
        if "gpcde" in name and "er60" in name:
            return "ID: GP ER60"
    if name.startswith("ood_graph_"):
        if "linear" in name:
            return "Graph: SBM + Linear"
        if "neuralnet" in name:
            return "Graph: SBM + MLP"
        if "gpcde" in name:
            return "Graph: SBM + GP"
    if name.startswith("ood_mech_"):
        if "periodic" in name:
            return "Mech: Periodic"
        if "square" in name:
            return "Mech: Square"
        if "logistic_map" in name:
            return "Mech: Logistic map"
        if "pnl_tanh" in name:
            return "Mech: PNL"
    if name.startswith("ood_noise_"):
        if "laplace" in name:
            return "Noise: Laplace"
        if "uniform" in name:
            return "Noise: Uniform"
    if name.startswith("ood_both_"):
        # Extract graph type: ood_both_<graph>_<mech>_d<nodes>_n<samples>
        graph_codes = {"sbm": "SBM", "ws": "WS", "grg": "GRG"}
        graph_label = "SBM"
        for code, label in graph_codes.items():
            if f"ood_both_{code}_" in name:
                graph_label = label
                break
        mech_codes = {
            "periodic": "Periodic",
            "logistic_map": "Logistic map",
            "pnl_tanh": "PNL",
        }
        for code, mech_label in mech_codes.items():
            if code in name:
                suffix = ""
                if "d60" in name:
                    # Include node/sample info for large-graph variants
                    d = name.split("_d")[-1].split("_")[0]
                    n = name.split("_n")[-1]
                    suffix = f" ({d},{n})"
                return f"Compound: {graph_label} + {mech_label}{suffix}"
    if name.startswith("ood_nodes_"):
        if "neuralnet" in name:
            return f"Nodes: MLP SF2 ({name.split('_d')[-1].split('_')[0]})"
        return f"Nodes: Linear SF2 ({name.split('_d')[-1].split('_')[0]})"
    if name.startswith("ood_samples_"):
        n_samples = name.split("_n")[-1]
        if "neuralnet" in name:
            return f"Samples: MLP SF2 ({n_samples})"
        return f"Samples: Linear ER20 ({n_samples})"
    return name


def generate_appendix_artifacts(thesis_root: Path, configs_root: Path) -> list[str]:
    from causal_meta.analysis.common.thesis import (GRAPHICS_APPENDIX_B,
                                                    GRAPHICS_APPENDIX_C)

    dir_b = thesis_root / GRAPHICS_APPENDIX_B
    dir_c = thesis_root / GRAPHICS_APPENDIX_C
    for d in (dir_b, dir_c):
        d.mkdir(parents=True, exist_ok=True)

    benchmark_cfg = _load_yaml(configs_root / "dg_2pretrain_multimodel.yaml")
    avici_cfg = _load_yaml(configs_root / "model" / "avici.yaml")
    bcnp_cfg = _load_yaml(configs_root / "model" / "bcnp.yaml")
    dibs_cfg = _load_yaml(configs_root / "inference" / "dibs" / "paper_faithful.yaml")
    bayesdag_cfg = _load_yaml(
        configs_root / "inference" / "bayesdag" / "paper_faithful.yaml"
    )
    random_cfg = _load_yaml(configs_root / "model" / "random.yaml")

    generated: list[str] = []

    model_rows = [
        [
            "AviCi",
            f"dim={avici_cfg.get('dim')}; layers={avici_cfg.get('layers')}; heads={avici_cfg.get('num_heads')}; dropout={avici_cfg.get('dropout')}",
        ],
        [
            "BCNP",
            f"d_model={bcnp_cfg.get('d_model')}; layers={bcnp_cfg.get('num_layers')}; decoder_layers={bcnp_cfg.get('num_layers_decoder')}; sinkhorn_iter={bcnp_cfg.get('sinkhorn_iter')}",
        ],
        [
            "DiBS",
            f"default_mode={dibs_cfg.get('mode')}; steps={dibs_cfg.get('steps')}; particles=d20:30,d50:10; timeout={dibs_cfg.get('external_timeout_s')}s",
        ],
        [
            "BayesDAG",
            f"variant={bayesdag_cfg.get('variant')}; chains={bayesdag_cfg.get('num_chains')}; max_epochs={bayesdag_cfg.get('max_epochs')}; timeout={bayesdag_cfg.get('external_timeout_s')}s",
        ],
        [
            "Random",
            f"randomize_topological_order={random_cfg.get('randomize_topological_order')}",
        ],
    ]
    _write_table(
        dir_b / "model_configurations_table.tex",
        caption="Model-specific configurations derived from the Hydra model and inference YAML files.",
        label="tab:appendix_model_configurations",
        headers=["Model", "Configuration summary"],
        rows=model_rows,
        colspec="lp{10cm}",
    )
    generated.append(f"{GRAPHICS_APPENDIX_B}/model_configurations_table.tex")

    val_rows: list[list[str]] = []
    for name, cfg in benchmark_cfg["data"].get("val_families", {}).items():
        val_rows.append(
            [
                _family_label(str(name)),
                _family_category(str(name)),
                _graph_summary(cfg.get("graph_cfg", {})),
                _mech_summary(cfg.get("mech_cfg", {})),
                _fmt(cfg.get("n_nodes")),
                _fmt(cfg.get("samples_per_task")),
            ]
        )
    _write_family_table(
        dir_c / "validation_families_table.tex",
        caption="Validation families used during benchmark training and selection.",
        label="tab:appendix_validation_families",
        rows=val_rows,
    )
    generated.append(f"{GRAPHICS_APPENDIX_C}/validation_families_table.tex")

    return generated
