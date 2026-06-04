#!/usr/bin/env bash
#SBATCH --job-name=cm_bcnp
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --tasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --time=168:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR_FROM_SCRIPT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${CAUSAL_META_ROOT_DIR:-${SLURM_SUBMIT_DIR:-${ROOT_DIR_FROM_SCRIPT}}}"
cd "${ROOT_DIR}"

CONFIG_NAME="${1:-dg_2pretrain_multimodel}"
if [[ "$#" -ge 1 ]]; then
  shift 1
fi

resolve_target_dir() {
  local raw_path="$1"
  if [[ "$raw_path" == /* ]]; then
    printf '%s\n' "$raw_path"
  elif [[ "${raw_path:0:2}" == "~/" ]]; then
    printf '%s\n' "${HOME}/${raw_path:2}"
  else
    printf '%s\n' "${ROOT_DIR}/${raw_path}"
  fi
}

EVAL_ONLY=false
RUN_NAME="bcnp_${SLURM_JOB_ID:-manual}"
TARGET_DIR="${ROOT_DIR}/experiments/runs/${RUN_NAME}"
PIPELINE_ARGS=()

if [[ "$#" -ge 1 && "$1" == "--eval-only" ]]; then
  if [[ "$#" -lt 2 ]]; then
    echo "Usage: $0 <config_name> --eval-only <target_run_dir> [hydra_overrides...]" >&2
    exit 1
  fi
  EVAL_ONLY=true
  TARGET_DIR="$(resolve_target_dir "$2")"
  RUN_NAME="$(basename "${TARGET_DIR}")"
  shift 2

  if [[ ! -d "${TARGET_DIR}" ]]; then
    echo "Eval-only target directory does not exist: ${TARGET_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${TARGET_DIR}/checkpoints/best.pt" ]]; then
    echo "Eval-only target is missing checkpoint: ${TARGET_DIR}/checkpoints/best.pt" >&2
    exit 1
  fi
else
  if [[ "$#" -ge 1 ]]; then
    RUN_NAME="$1"
    shift 1
  fi
  TARGET_DIR="${ROOT_DIR}/experiments/runs/${RUN_NAME}"
  mkdir -p "${TARGET_DIR}"
fi

LOG_FILE="${TARGET_DIR}/slurm_${SLURM_JOB_ID:-manual}.log"
exec >>"${LOG_FILE}" 2>&1

VENV_DIR="${VENV_DIR:-${UV_PROJECT_ENVIRONMENT:-.venv}}"
if [[ "${VENV_DIR}" != /* ]]; then
  VENV_DIR="${ROOT_DIR}/${VENV_DIR}"
fi

MAIN_PYTHON="${VENV_DIR}/bin/python"
if [[ ! -x "${MAIN_PYTHON}" ]]; then
  MAIN_PYTHON="python3"
fi

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)}"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
export MASTER_ADDR

job_id_for_port="${SLURM_JOB_ID:-0}"
MASTER_PORT="${MASTER_PORT:-$((10000 + (job_id_for_port % 50000)))}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HYDRA_FULL_ERROR=1
export PYTHONFAULTHANDLER=1
export MASTER_PORT

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-unset}"
echo "=== scontrol show job ${SLURM_JOB_ID:-unset} ==="
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  scontrol show job "${SLURM_JOB_ID}" || true
fi
echo "=== scontrol show node ${SLURMD_NODENAME:-${HOSTNAME:-unset}} ==="
if [[ -n "${SLURMD_NODENAME:-}" ]]; then
  scontrol show node "${SLURMD_NODENAME}" || true
fi

if [[ "${EVAL_ONLY}" == "true" ]]; then
  echo "Running in eval-only mode for ${TARGET_DIR}"
  rm -f "${TARGET_DIR}/metrics.json"
  rm -rf "${TARGET_DIR}/inference"
  PIPELINE_ARGS=(
    "++inference.output_dir=${TARGET_DIR}"
    "++inference.use_best_checkpoint_for_eval=true"
  )
fi

echo "Launching ${SLURM_NTASKS:-2} tasks via srun"

srun "${MAIN_PYTHON}" -m causal_meta.main \
  --config-name "${CONFIG_NAME}" \
  "model=bcnp" \
  "$@" \
  "name=${RUN_NAME}" \
  "${PIPELINE_ARGS[@]}"
