#!/bin/bash
# Resources are supplied by the submitter from the frozen experiment config.
set -euo pipefail
export PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${PROJECT_ROOT}/ops/slurm/common.sh"
mc_setup_python_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

# Slurm may release this consumer immediately after freeze_choices exits while
# another compute node still has stale NFS metadata for its status file. The
# afterok dependency already guarantees ordering; this short interval only
# gives the shared filesystem time to expose the durable completion marker.
if [ "$2" = "endpoint_chunk" ]; then
  sleep "${MC_NFS_SETTLE_SECONDS:-15}"
fi

srun python -m experiments.dense_linear_stage.run "$1" "$2" "$3"
