#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_NAME="${1:-dg_2pretrain_multimodel}"
if [[ "$#" -ge 1 ]]; then
  shift 1
fi

if [[ "$#" -ge 1 && "${1}" != *=* ]]; then
  RUN_PREFIX="${1}"
  shift 1
else
  RUN_PREFIX="ablation_$(date +%Y%m%d_%H%M%S)"
fi

MODELS_CSV="${CAUSAL_META_ABLATION_MODELS:-avici,bcnp}"
DATA_CONFIGS_CSV="${CAUSAL_META_ABLATION_DATA_CONFIGS:-ablation_linear_only,ablation_linear_mlp,ablation_er_only,ablation_er_sf_full}"

IFS=',' read -r -a MODELS <<< "${MODELS_CSV}"
IFS=',' read -r -a DATA_CONFIGS <<< "${DATA_CONFIGS_CSV}"

EXTRA_OVERRIDES=("$@")

script_for_model() {
  case "$1" in
    avici) echo "run_avici.sh" ;;
    bcnp) echo "run_bcnp.sh" ;;
    dibs) echo "run_dibs.sh" ;;
    bayesdag) echo "run_bayesdag.sh" ;;
    *)
      echo ""
      ;;
  esac
}

submitted=0
for data_cfg in "${DATA_CONFIGS[@]}"; do
  for model in "${MODELS[@]}"; do
    script_name="$(script_for_model "${model}")"
    if [[ -z "${script_name}" ]]; then
      echo "Skipping unknown model '${model}'" >&2
      continue
    fi

    run_name="${RUN_PREFIX}_${data_cfg}_${model}"
    job_id="$(
      sbatch \
        --parsable \
        --chdir "${ROOT_DIR}" \
        --export="ALL,CAUSAL_META_ROOT_DIR=${ROOT_DIR}" \
        "${SCRIPT_DIR}/${script_name}" \
        "${CONFIG_NAME}" \
        "${run_name}" \
        "+data=${data_cfg}" \
        "${EXTRA_OVERRIDES[@]}"
    )"
    echo "Submitted model=${model} data=${data_cfg} job_id=${job_id} run_name=${run_name}"
    submitted=$((submitted + 1))
  done
done

echo "Submission complete. Jobs submitted=${submitted} prefix=${RUN_PREFIX} config=${CONFIG_NAME}"
