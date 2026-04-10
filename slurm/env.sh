#!/usr/bin/env bash
# Source this file at the top of every SLURM script and in interactive
# login sessions on the cluster. It loads the aarch64 GH200 toolchain,
# activates the project virtualenv, and exports path variables used by
# the training code.
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

# $SCRATCH is the conventional per user fast storage on HPC sites. Fall
# back to a local directory for development on a laptop.
: "${SCRATCH:=${HOME}/scratch}"
export GEMMA4_PL_ROOT="${GEMMA4_PL_ROOT:-${SCRATCH}/gemma4-pl}"
export GEMMA4_PL_CACHE="${GEMMA4_PL_CACHE:-${GEMMA4_PL_ROOT}/cache}"
export GEMMA4_PL_MODELS="${GEMMA4_PL_MODELS:-${GEMMA4_PL_ROOT}/models}"
export GEMMA4_PL_DATA="${GEMMA4_PL_DATA:-${GEMMA4_PL_ROOT}/data}"
export GEMMA4_PL_CHECKPOINTS="${GEMMA4_PL_CHECKPOINTS:-${GEMMA4_PL_ROOT}/checkpoints}"
mkdir -p "${GEMMA4_PL_CACHE}" "${GEMMA4_PL_MODELS}" "${GEMMA4_PL_DATA}" "${GEMMA4_PL_CHECKPOINTS}"

# HuggingFace and Transformers caches on SCRATCH (never on $HOME, quota)
export HF_HOME="${GEMMA4_PL_CACHE}/huggingface"
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
VENV_PATH="${VENV_PATH:-${GEMMA4_PL_ROOT}/venvs/gemma4-pl}"
if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck disable=SC1090,SC1091
    source "${VENV_PATH}/bin/activate"
    export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
fi

# Load secrets and site overrides if present
if [[ -f "${PWD}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${PWD}/.env"
    set +a
fi

echo "[env.sh] Modules loaded, GEMMA4_PL_ROOT=${GEMMA4_PL_ROOT}"
