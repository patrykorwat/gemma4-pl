#!/usr/bin/env bash
# Source this file at the top of every SLURM script and in interactive
# login sessions on Helios. It loads the aarch64 GH200 toolchain, activates
# the project virtualenv, and exports path variables used by the training
# code.
#
# Usage:
#   source slurm/env.sh
#
# This script is idempotent, it will not double load modules.

set -euo pipefail

# Purge only if we are inside a non trivial module stack, keep login defaults.
if command -v module >/dev/null 2>&1; then
    module purge
    module load GCC/13.2.0
    module load Python/3.11.5
    module load CUDA/12.9.1
    module load cuDNN/9.19.1.2-CUDA-12.9.1
    module load NVHPC/25.9-CUDA-12.9.1
fi

# Paths. SCRATCH is PLGrid convention. Fall back to a local dir for dev.
: "${SCRATCH:=${HOME}/scratch}"
export BIELIK_R_ROOT="${BIELIK_R_ROOT:-${SCRATCH}/bielik-r}"
export BIELIK_R_CACHE="${BIELIK_R_CACHE:-${BIELIK_R_ROOT}/cache}"
export BIELIK_R_MODELS="${BIELIK_R_MODELS:-${BIELIK_R_ROOT}/models}"
export BIELIK_R_DATA="${BIELIK_R_DATA:-${BIELIK_R_ROOT}/data}"
export BIELIK_R_CHECKPOINTS="${BIELIK_R_CHECKPOINTS:-${BIELIK_R_ROOT}/checkpoints}"
mkdir -p "${BIELIK_R_CACHE}" "${BIELIK_R_MODELS}" "${BIELIK_R_DATA}" "${BIELIK_R_CHECKPOINTS}"

# HuggingFace and Transformers caches on SCRATCH (never on $HOME, quota)
export HF_HOME="${BIELIK_R_CACHE}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${HF_HUB_CACHE}"

# Torch and NCCL tuning for GH200 NVLink
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_NET_GDR_LEVEL=PHB
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Threading
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Activate the project venv if it exists
VENV_PATH="${VENV_PATH:-${BIELIK_R_ROOT}/venvs/bielik-r}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "${VENV_PATH}/bin/activate"
    export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
fi

# Load secrets if present (HF_TOKEN, WANDB_API_KEY, PLG_GRANT)
if [[ -f "${PWD}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PWD}/.env"
    set +a
fi

echo "[env.sh] Modules loaded, BIELIK_R_ROOT=${BIELIK_R_ROOT}"
