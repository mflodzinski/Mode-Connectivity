#!/bin/bash
set -euo pipefail

OPERATION="$1"
shift

export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/ops/slurm/common.sh"
mc_setup_python_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

srun python -m experiments.training_stage.alignment_grid_evaluation "${OPERATION}" "$@"
