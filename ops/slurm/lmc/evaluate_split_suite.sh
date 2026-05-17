#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_eval_suite_%x_%j.out
#SBATCH --error=slurm_lmc_eval_suite_%x_%j.err
#SBATCH --job-name=lmc_eval_suite
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

mc_setup_python_env
mc_banner "Evaluate LMC Split Suite"

args=(
  --data-root "${DATA_ROOT:-./data}"
  --batch-size "${BATCH_SIZE:-128}"
  --workers "${NUM_WORKERS:-2}"
  --num-points "${NUM_POINTS:-61}"
)
if [ "$#" -gt 0 ]; then
  args+=(--labels "$@")
fi

mc_run_module experiments.lmc.evaluate_pytorch_vgg_split_suite "${args[@]}"
