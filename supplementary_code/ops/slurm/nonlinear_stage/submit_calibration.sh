#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
mc_setup_python_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m experiments.nonlinear_stage.submit \
  --mode calibration \
  output_root=results/nonlinear_stage_vgg11_calibration \
  'source_pairs=[{seeds:[0,1],root:results/training_stage_vgg11_pair01_atol2e5}]' \
  "$@"
