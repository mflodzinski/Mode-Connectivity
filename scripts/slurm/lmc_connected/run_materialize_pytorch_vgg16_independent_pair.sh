#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_pytorch_vgg_independent_%x_%j.out
#SBATCH --error=slurm_pytorch_vgg_independent_%x_%j.err
#SBATCH --job-name=pytorch_vgg_independent

set -euo pipefail

source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:${PYTHONPATH:-}"

echo "========================================"
echo "Package existing PyTorch VGG16 independent pair"
echo "========================================"
echo ""

srun python scripts/train/materialize_pytorch_vgg16_independent_pair.py

echo ""
echo "========================================"
echo "Evaluate independent pair interpolation"
echo "========================================"
echo ""

srun python scripts/analysis/evaluate_pytorch_vgg_pair.py \
  --w0 results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing/seed0/checkpoint-200.pt \
  --w1 results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing/seed1/checkpoint-200.pt \
  --data-root ./data \
  --batch-size 128 \
  --workers 4 \
  --num-points 61 \
  --output-dir results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing/evaluation
