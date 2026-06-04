#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE="${1:-smoke}"
if [[ "$#" -ge 1 ]]; then
  shift 1
fi

if [[ "${STAGE}" == "smoke" || "${STAGE}" == "canary" ]]; then
  CONFIG_NAME="dg_2pretrain_smoke"
elif [[ "${STAGE}" == "main" || "${STAGE}" == "full" ]]; then
  CONFIG_NAME="dg_2pretrain_multimodel"
else
  echo "Usage: $0 [smoke|main] [run_prefix] [hydra_overrides...]" >&2
  exit 1
fi

if [[ "$#" -ge 1 && "${1}" != *=* ]]; then
  RUN_PREFIX="${1}"
  shift 1
else
  RUN_PREFIX="${STAGE}_$(date +%Y%m%d_%H%M%S)"
fi

EXTRA_OVERRIDES=("$@")

submit_one() {
  local script_name="$1"
  local model_name="$2"
  local run_name="${RUN_PREFIX}_${model_name}"

  local job_id
  job_id="$(
    sbatch \
      --parsable \
      --chdir "${ROOT_DIR}" \
      --export="ALL,CAUSAL_META_ROOT_DIR=${ROOT_DIR}" \
      "${SCRIPT_DIR}/${script_name}" \
      "${CONFIG_NAME}" \
      "${run_name}" \
      "${EXTRA_OVERRIDES[@]}"
  )"
  echo "Submitted ${model_name}: job_id=${job_id} run_name=${run_name} config=${CONFIG_NAME}"
}

submit_one "run_avici.sh" "avici"
submit_one "run_bcnp.sh" "bcnp"
submit_one "run_dibs.sh" "dibs"
submit_one "run_bayesdag.sh" "bayesdag"

echo "Submission complete. Prefix=${RUN_PREFIX}"
