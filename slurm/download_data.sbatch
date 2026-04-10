#!/usr/bin/env bash
#SBATCH --job-name=gemma4-pl-download
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-download-%j.out
#SBATCH --error=logs/slurm-download-%j.err

# Pass --account and --partition on the sbatch command line:
#
#   sbatch --account=<acct> --partition=<cpu-partition> slurm/download_data.sbatch

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-${PWD}}"
source slurm/env.sh

echo "[$(date -Is)] Downloading base model and datasets"
python scripts/download_base_model.py \
    --repo "${BASE_MODEL:-google/gemma-4-E4B}" \
    --dest "${GEMMA4_PL_MODELS}"

python scripts/download_datasets.py \
    --dest "${GEMMA4_PL_DATA}"

echo "[$(date -Is)] Done"
