#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_sinkhorn_three_way_%x_%j.out
#SBATCH --error=slurm_sinkhorn_three_way_%x_%j.err
#SBATCH --job-name=sinkhorn_3way
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

mc_setup_python_env
mc_require_external_file "external/sinkhorn-rebasin/examples/models/vgg.py"
mc_banner "Sinkhorn Three-Way Comparison"

args=(
  --vgg-name "${VGG_NAME:-VGG13}"
  --model-a-checkpoint "${MODEL_A_CHECKPOINT:-external/pytorch-vgg-cifar10/save_vgg13_seed0/model_final_state_dict.pth}"
  --model-b-checkpoint "${MODEL_B_CHECKPOINT:-external/pytorch-vgg-cifar10/save_vgg13_seed1/model_final_state_dict.pth}"
  --rebased-perm-checkpoint "${REBASED_PERM_CHECKPOINT:-results/vgg13/cifar10/raw_pth_align_sweep_joint_permutation_cor_def/steps150_tau1p0_lr0p75_l1p0_lossmidpoint/rebased_model.pt}"
  --rebased-scale-checkpoint "${REBASED_SCALE_CHECKPOINT:-results/vgg13/cifar10/raw_pth_align_sweep_joint_scale_cor_def/steps150_tau2p5_lr0p05_l1p0_lossmidpoint_lam0p003/rebased_model.pt}"
  --output-dir "${OUTPUT_DIR:-results/vgg13/cifar10/interpolation_comparison_three_way}"
  --data-path "${DATA_PATH:-./data}"
  --image-size "${IMAGE_SIZE:-32}"
  --batch-size "${BATCH_SIZE:-1000}"
  --num-workers "${NUM_WORKERS:-4}"
  --num-eval-points "${NUM_EVAL_POINTS:-51}"
  --device "${DEVICE:-cuda}"
)
if [ "${SKIP_PLOTS:-true}" = "true" ]; then
  args+=(--skip-plots)
fi

mc_run_module experiments.sinkhorn.vgg_cifar_three_way_comparison "${args[@]}"
