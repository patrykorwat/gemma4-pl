#!/usr/bin/env bash
#SBATCH --job-name=gemma4-pl-sft
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm-sft-%j.out
#SBATCH --error=logs/slurm-sft-%j.err

# Account and partition are not hard coded. Pass them on the sbatch
# command line so the script stays site agnostic:
#
#   sbatch --account=<acct> --partition=<gpu-partition> slurm/sft.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
source slurm/env.sh

export MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=${SLURM_NTASKS_PER_NODE:-4}

echo "[$(date -Is)] Launching SFT on ${WORLD_SIZE} GPUs, master ${MASTER_ADDR}:${MASTER_PORT}"

accelerate launch \
    --num_processes "${WORLD_SIZE}" \
    --num_machines 1 \
    --mixed_precision bf16 \
    --dynamo_backend no \
    -m gemma4_pl.training.sft \
    --config config/sft.yaml \
    --output_dir "${GEMMA4_PL_CHECKPOINTS}/sft"

echo "[$(date -Is)] SFT finished"
