#!/bin/bash
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PROJECT_ROOT}/ops/slurm/common.sh"
mc_setup_python_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

operation="$1"
experiment_root="$2"
if [ "${operation}" = "evaluate" ]; then
  srun python -m experiments.dense_linear_stage.full_train \
    evaluate "${experiment_root}" "${SLURM_ARRAY_TASK_ID}"
else
  srun python -m experiments.dense_linear_stage.full_train \
    "${operation}" "${experiment_root}"
fi
