#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_verify_equivalence_%j.out
#SBATCH --error=slurm_verify_equivalence_%j.err
#SBATCH --job-name=verify_equiv
#SBATCH --gres=gpu:a40:1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../common.sh"

mc_setup_python_env
mc_require_external_file "external/sinkhorn-rebasin/examples/models/vgg.py"
mc_banner "Check Model Functional Equivalence"

mc_run_module tools.verification.check_model_functional_equivalence \
  --model-a-checkpoint "${MODEL_A_CHECKPOINT:-VGG11_cifar10_0.911.pth}" \
  --rebased-checkpoint "${REBASED_CHECKPOINT:-results/vgg11/cifar10/raw_pth_align_sweep_dist_l2_perm_only/steps150_tau1p0_lr0p01_l1p0_lossdist_l2/rebased_model.pt}" \
  --vgg-name "${VGG_NAME:-VGG11}" \
  --num-classes "${NUM_CLASSES:-10}" \
  --data-path "${DATA_PATH:-./data}" \
  --image-size "${IMAGE_SIZE:-32}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --num-workers "${NUM_WORKERS:-4}" \
  --device "${DEVICE:-cuda}" \
  --output-json "${OUTPUT_JSON:-results/vgg11/cifar10/functional_equivalence.json}"
