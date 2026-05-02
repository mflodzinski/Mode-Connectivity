#!/bin/bash

set -euo pipefail

SPLIT_ITERS=(0 25 100 1000 5000)

echo "Submitting shared trunk job..."
SHARED_JOB_ID=$(
  sbatch scripts/slurm/lmc_connected/run_vgg16_lmc_iter_noaug_shared.sh \
    | awk '{print $4}'
)
echo "  shared job id: ${SHARED_JOB_ID}"

for SPLIT_ITER in "${SPLIT_ITERS[@]}"; do
  echo "Submitting pair job for split ${SPLIT_ITER}..."
  PAIR_JOB_ID=$(
    sbatch \
      --dependency=afterok:${SHARED_JOB_ID} \
      --job-name="lmc_iter_pair_${SPLIT_ITER}" \
      scripts/slurm/lmc_connected/run_vgg16_lmc_iter_noaug_pair.sh "${SPLIT_ITER}" \
      | awk '{print $4}'
  )
  echo "  pair job id: ${PAIR_JOB_ID}"

  echo "Submitting benchmark job for split ${SPLIT_ITER}..."
  BENCH_JOB_ID=$(
    sbatch \
      --dependency=afterok:${PAIR_JOB_ID} \
      --job-name="lmc_iter_bench_${SPLIT_ITER}" \
      scripts/slurm/lmc_connected/run_vgg16_lmc_iter_noaug_benchmark.sh "${SPLIT_ITER}" \
      | awk '{print $4}'
  )
  echo "  benchmark job id: ${BENCH_JOB_ID}"
done

echo ""
echo "Pipeline submitted."
echo "Shared job: ${SHARED_JOB_ID}"
echo "Split iterations: ${SPLIT_ITERS[*]}"
