#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STAGE="${1:-smoke}"

if [[ "${STAGE}" == "smoke" || "${STAGE}" == "canary" ]]; then
  CONFIG_NAME="dg_2pretrain_smoke"
elif [[ "${STAGE}" == "main" || "${STAGE}" == "full" ]]; then
  CONFIG_NAME="dg_2pretrain_multimodel"
else
  echo "Usage: $0 [smoke|main]" >&2
  exit 1
fi

RUN_PREFIX="${STAGE}_$(date +%Y%m%d_%H%M%S)"

"${SCRIPT_DIR}/run_avici.sh" "${CONFIG_NAME}" "${RUN_PREFIX}_avici"
"${SCRIPT_DIR}/run_bcnp.sh" "${CONFIG_NAME}" "${RUN_PREFIX}_bcnp"
"${SCRIPT_DIR}/run_dibs.sh" "${CONFIG_NAME}" "${RUN_PREFIX}_dibs"
"${SCRIPT_DIR}/run_bayesdag.sh" "${CONFIG_NAME}" "${RUN_PREFIX}_bayesdag"
