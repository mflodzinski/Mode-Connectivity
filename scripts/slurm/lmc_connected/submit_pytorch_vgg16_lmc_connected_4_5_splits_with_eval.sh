#!/bin/bash
set -euo pipefail

TRAIN_SCRIPT="scripts/slurm/lmc_connected/run_pytorch_vgg16_lmc_connected_from_scratch.sh"
EVAL_SCRIPT="scripts/slurm/analysis/run_pytorch_vgg_split_pair_eval.sh"

submit_pair() {
  local config_name="$1"
  local label="$2"
  local train_job
  train_job=$(sbatch --parsable "${TRAIN_SCRIPT}" "${config_name}")
  echo "Submitted training ${config_name}: ${train_job}"
  sbatch --dependency=afterok:"${train_job}" "${EVAL_SCRIPT}" "${label}"
}

submit_pair "vgg16_lmc_connected_pair_4split" "4/196"
submit_pair "vgg16_lmc_connected_pair_5split" "5/195"
