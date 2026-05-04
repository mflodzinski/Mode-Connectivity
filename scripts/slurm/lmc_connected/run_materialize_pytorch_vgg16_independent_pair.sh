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
