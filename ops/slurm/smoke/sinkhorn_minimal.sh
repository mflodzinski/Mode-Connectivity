#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_smoke_sinkhorn_%j.out
#SBATCH --error=slurm_smoke_sinkhorn_%j.err
#SBATCH --job-name=smoke_sinkhorn
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
COMMON_SH="${SUBMIT_DIR}/ops/slurm/common.sh"
# shellcheck disable=SC1090
source "${COMMON_SH}"

mc_setup_python_env
mc_require_external_file "external/sinkhorn-rebasin/examples/models/vgg.py"
mc_banner "Smoke Check: Sinkhorn Alignment"

BASE_OUTPUT_ROOT="${BASE_OUTPUT_ROOT:-results/smoke/sinkhorn/vgg16_perm_only_50iters}"
MODEL_A_CHECKPOINT="${MODEL_A_CHECKPOINT:-external/pytorch-vgg-cifar10/save_vgg16_seed0/model_final_state_dict.pth}"
MODEL_B_CHECKPOINT="${MODEL_B_CHECKPOINT:-external/pytorch-vgg-cifar10/save_vgg16_seed1/model_final_state_dict.pth}"
VGG_NAME="${VGG_NAME:-VGG16}"

echo "BASE_OUTPUT_ROOT: ${BASE_OUTPUT_ROOT}"
echo "MODEL_A_CHECKPOINT: ${MODEL_A_CHECKPOINT}"
echo "MODEL_B_CHECKPOINT: ${MODEL_B_CHECKPOINT}"
echo "VGG_NAME: ${VGG_NAME}"
echo ""

mc_run_module experiments.sinkhorn.vgg_cifar_alignment_sweep \
  ++base_output_root="${BASE_OUTPUT_ROOT}" \
  ++model_a_checkpoint="${MODEL_A_CHECKPOINT}" \
  ++model_b_checkpoint="${MODEL_B_CHECKPOINT}" \
  ++vgg_name="${VGG_NAME}" \
  ++start_index=0 \
  ++end_index=0 \
  ++continue_on_error=false \
  ++num_workers="${SLURM_CPUS_PER_TASK}" \
  ++best_eval_interval=5 \
  ++early_stopping_patience=2 \
  '++validation_alpha_grid=[0.5]' \
  '++sweep.alignment_iterations=[50]' \
  '++sweep.loss_name=[dist_l2]' \
  '++sweep.tau=[1.0]' \
  '++sweep.lr=[0.01]' \
  '++sweep.sinkhorn_l=[1.0]'
