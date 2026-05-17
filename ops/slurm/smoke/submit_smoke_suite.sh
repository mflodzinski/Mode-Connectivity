#!/bin/bash

set -euo pipefail

submit_job() {
  local script_path="$1"
  shift
  sbatch --parsable "${script_path}" "$@"
}

echo "Submitting independent smoke jobs..."
curve_job=$(submit_job ops/slurm/smoke/curve_minimal.sh)
sinkhorn_job=$(submit_job ops/slurm/smoke/sinkhorn_minimal.sh)
lmc_resume_job=$(submit_job ops/slurm/smoke/lmc_resume_minimal.sh)
xor_linear_job=$(submit_job ops/slurm/smoke/xor_train_linear_minimal.sh)

echo "curve_job=${curve_job}"
echo "sinkhorn_job=${sinkhorn_job}"
echo "lmc_resume_job=${lmc_resume_job}"
echo "xor_linear_job=${xor_linear_job}"

echo ""
echo "Submitting dependent smoke jobs..."
lmc_benchmark_job=$(sbatch --parsable --dependency=afterok:"${lmc_resume_job}" ops/slurm/smoke/lmc_benchmark_minimal.sh)
xor_perm_scale_job=$(sbatch --parsable --dependency=afterok:"${xor_linear_job}" ops/slurm/smoke/xor_permutation_scale_minimal.sh)

echo "lmc_benchmark_job=${lmc_benchmark_job}"
echo "xor_perm_scale_job=${xor_perm_scale_job}"
