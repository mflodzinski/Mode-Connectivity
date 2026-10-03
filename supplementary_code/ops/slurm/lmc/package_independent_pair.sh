#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_independent_%x_%j.out
#SBATCH --error=slurm_lmc_independent_%x_%j.err
#SBATCH --job-name=lmc_independent
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_banner "Package Independent Pair"
mc_run_module experiments.lmc.materialize_pytorch_vgg16_independent_pair

mc_banner "Evaluate Independent Pair"
mc_run_module experiments.lmc.evaluate_pytorch_vgg_pair \
  --w0 results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing/seed0/checkpoint-200.pt \
  --w1 results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing/seed1/checkpoint-200.pt \
  --data-root "${DATA_ROOT:-./data}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --workers "${NUM_WORKERS:-2}" \
  --num-points "${NUM_POINTS:-61}" \
  --output-dir results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing/evaluation
