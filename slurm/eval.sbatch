#!/usr/bin/env bash
#SBATCH --job-name=gemma4-pl-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm-eval-%j.out
#SBATCH --error=logs/slurm-eval-%j.err

# Pass --account and --partition on the sbatch command line:
#
#   sbatch --account=<acct> --partition=<gpu-partition> slurm/eval.sbatch [checkpoint] [suite]

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
source slurm/env.sh

CHECKPOINT="${1:-${GEMMA4_PL_CHECKPOINTS}/sft}"
SUITE="${2:-polish}"

echo "[$(date -Is)] Evaluating ${CHECKPOINT} on suite ${SUITE}"

python scripts/run_eval.py \
    --checkpoint "${CHECKPOINT}" \
    --suite "${SUITE}" \
    --output "logs/eval-${SLURM_JOB_ID}.json"

echo "[$(date -Is)] Eval finished"
