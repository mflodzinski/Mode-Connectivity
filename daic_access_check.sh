#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:02:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --job-name=daic_check
#SBATCH --output=daic_check_%j.out
#SBATCH --error=daic_check_%j.err

set -euo pipefail

echo "DAIC Slurm access check"
echo "date: $(date)"
echo "user: $(whoami)"
echo "host: $(hostname)"
echo "submit dir: ${SLURM_SUBMIT_DIR:-unset}"
echo "job id: ${SLURM_JOB_ID:-unset}"
echo "partition: ${SLURM_JOB_PARTITION:-unset}"

srun hostname
