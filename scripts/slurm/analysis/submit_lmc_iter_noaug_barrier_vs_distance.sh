#!/bin/bash

set -euo pipefail

METRICS=(
  test_loss_barrier_rel
  train_loss_barrier_rel
  test_acc_barrier_rel
  train_acc_barrier_rel
)

for METRIC in "${METRICS[@]}"; do
  echo "Submitting plot job for ${METRIC}..."
  JOB_ID=$(
    sbatch \
      --job-name="lmc_iter_plot_${METRIC}" \
      scripts/slurm/analysis/run_lmc_iter_noaug_barrier_vs_distance.sh "${METRIC}" \
      | awk '{print $4}'
  )
  echo "  job id: ${JOB_ID}"
done

echo ""
echo "Submitted metrics: ${METRICS[*]}"
