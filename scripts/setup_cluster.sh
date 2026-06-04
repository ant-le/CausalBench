#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

"${SCRIPT_DIR}/bootstrap_uv.sh"
uv sync --extra cluster --extra wandb --frozen --no-editable
uv pip install --python .venv-bayesdag/bin/python -r requirements-bayesdag.txt

echo "Cluster setup complete."
echo "Main env:     ${ROOT_DIR}/.venv/bin/python"
echo "BayesDAG env: ${ROOT_DIR}/.venv-bayesdag/bin/python"
