#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_random_plane_codim_%j.out
#SBATCH --error=slurm_random_plane_codim_%j.err
#SBATCH --job-name=rpm_codim
#SBATCH --gres=gpu:a40:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity}"
CONFIG_NAME="${CONFIG_NAME:-vgg16_randomplane_midpoint_seed0-seed1}"
CODIM="${CODIM:?Set CODIM, e.g. 5 or 10}"
EPOCHS="${EPOCHS:-100}"
TRAIN_HALF_ONLY="${TRAIN_HALF_ONLY:-true}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for this run}"

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/external/dnn-mode-connectivity${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/evaluations"

echo "========================================"
echo "Random-Plane Codimension No-Aug Run"
echo "========================================"
echo "CONFIG_NAME: $CONFIG_NAME"
echo "CODIM: $CODIM"
echo "EPOCHS: $EPOCHS"
echo "TRAIN_HALF_ONLY: $TRAIN_HALF_ONLY"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "========================================"

echo "STEP 1: Training"
srun python scripts/train/run_random_plane.py \
  --config-name "$CONFIG_NAME" \
  ++random_plane_codim="$CODIM" \
  ++no_train_aug=true \
  ++train_half_only="$TRAIN_HALF_ONLY" \
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
