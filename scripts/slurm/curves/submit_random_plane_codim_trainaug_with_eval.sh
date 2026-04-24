#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=3GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_random_plane_codim_trainaug_%j.out
#SBATCH --error=slurm_random_plane_codim_trainaug_%j.err
#SBATCH --job-name=rpm_codim_aug
#SBATCH --gres=gpu:a40:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity}"
CONFIG_NAME="${CONFIG_NAME:-vgg16_randomplane_midpoint_seed0-seed1}"
CODIM="${CODIM:?Set CODIM, e.g. 1, 5, 10, 30}"
EPOCHS="${EPOCHS:-100}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for this run}"

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/external/dnn-mode-connectivity${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/evaluations"

echo "========================================"
echo "Random-Plane Codimension Train-Aug Full-Path Run"
echo "========================================"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "CODIM: $CODIM"
echo "EPOCHS: $EPOCHS"
echo "NO_TRAIN_AUG: false"
echo "TRAIN_HALF_ONLY: false"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "========================================"

echo "STEP 1: Training"
srun python scripts/train/run_random_plane.py \
  --config-name "$CONFIG_NAME" \
  ++random_plane_codim="$CODIM" \
  ++no_train_aug=false \
  ++train_half_only=false \
  epochs="$EPOCHS" \
  output_dir="$OUTPUT_DIR"

echo "STEP 2: Evaluation"
srun python external/dnn-mode-connectivity/eval_curve.py \
  --dir "$OUTPUT_DIR/evaluations" \
  --dataset CIFAR10 \
  --use_test \
  --transform VGG \
  --data_path ./data \
  --model VGG16 \
  --curve PolyChain \
  --num_bends 3 \
  --ckpt "$OUTPUT_DIR/checkpoint-${EPOCHS}.pt" \
  --num_points 61

echo "TRAINING AND EVALUATION COMPLETED"
