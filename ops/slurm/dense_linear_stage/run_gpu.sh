#!/bin/bash
# Resources are supplied by the submitter from the frozen experiment config.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PROJECT_ROOT}/ops/slurm/common.sh"
mc_setup_python_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
srun python -m experiments.dense_linear_stage.run "$1" "$2" "$3"

