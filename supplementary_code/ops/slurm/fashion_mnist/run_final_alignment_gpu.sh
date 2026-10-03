#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:a40:1
#SBATCH --signal=USR1@120

set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PROJECT_ROOT}/ops/slurm/common.sh"
mc_setup_python_env
mc_require_external_file "external/sinkhorn-rebasin/rebasin/rebasinnet/symmnet.py"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

manifest="$1"
operation="$2"
replicate="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
srun python -m experiments.fashion_mnist.final_alignment_run \
  "${operation}" --manifest "${manifest}" --replicate "${replicate}"
