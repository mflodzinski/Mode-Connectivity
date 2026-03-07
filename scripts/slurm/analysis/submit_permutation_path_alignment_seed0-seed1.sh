#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_permutation_path_alignment_%j.out
#SBATCH --error=slurm_permutation_path_alignment_%j.err
#SBATCH --job-name=perm_path_align
#SBATCH --gres=gpu:a40:1

source "$HOME/venvs/mode-connectivity/bin/activate" || . "$HOME/venvs/mode-connectivity/bin/activate"

cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

mkdir -p results/vgg16/cifar10/alignment/permutation_path

srun bash scripts/experiments/run_permutation_path_alignment.sh \
  results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt \
  results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt \
  results/vgg16/cifar10/alignment/permutation_path \
  seed0-seed1_polychain_path_alignment
