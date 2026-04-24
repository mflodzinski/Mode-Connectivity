#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_curve_trainaug_%j.out
#SBATCH --error=slurm_curve_trainaug_%j.err
#SBATCH --job-name=curve_trainaug
#SBATCH --gres=gpu:a40:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity}"
CONFIG_NAME="${CONFIG_NAME:?Set CONFIG_NAME, e.g. vgg16_curve_seed1-randperm_reg}"
OUTPUT_ROOT="${OUTPUT_ROOT:?Set OUTPUT_ROOT, e.g. results/vgg16/cifar10/curves/standard_trainaug/seed1-randperm_reg/checkpoints}"

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_ROOT"
mkdir -p "${OUTPUT_ROOT/checkpoints/evaluations}"
mkdir -p "${OUTPUT_ROOT/checkpoints/figures}"

echo "========================================"
echo "Garipov Bezier Curve With Train Augmentation"
echo "========================================"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "========================================"

echo "STEP 1: Training"
srun python scripts/train/run_garipov_curve.py \
  --config-name "$CONFIG_NAME" \
  no_train_aug=false \
  output_root="$OUTPUT_ROOT"

echo "STEP 2: Evaluation"
srun python scripts/eval/eval_garipov_curve.py \
  --config-name "$CONFIG_NAME" \
  no_train_aug=false \
  output_root="$OUTPUT_ROOT"

echo "TRAINING AND EVALUATION COMPLETED"
