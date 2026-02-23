#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:45:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_create_seed1_randperm_%j.out
#SBATCH --error=slurm_create_seed1_randperm_%j.err
#SBATCH --job-name=create_seed1_randperm
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root and scripts directory to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/scripts:$PYTHONPATH

# Create output directories
mkdir -p results/vgg16/cifar10/endpoints/standard/seed1_randperm/checkpoints
mkdir -p results/vgg16/cifar10/endpoints/standard/seed1_randperm/evaluations

echo ""
echo "========================================"
echo "STEP 1: Creating Random-Permuted Checkpoint"
echo "========================================"
srun python scripts/analysis/network_transform.py \
  --mode random \
  --checkpoint results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt \
  --output results/vgg16/cifar10/endpoints/standard/seed1_randperm/checkpoints/checkpoint-200.pt \
  --model VGG16 \
  --perm-seed 42 \
  --verify \
  --full-dataset-verify \
  --dataset CIFAR10 \
  --data-path ./data \
  --batch-size 128 \
  --num-workers 4

if [ $? -ne 0 ]; then
    echo "Random permutation creation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "STEP 2: Evaluating Linear Interpolation"
echo "========================================"
srun python scripts/eval/evaluate.py \
    --mode linear \
    --dir results/vgg16/cifar10/endpoints/standard/seed1_randperm/evaluations \
    --init-start results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt \
    --init-end results/vgg16/cifar10/endpoints/standard/seed1_randperm/checkpoints/checkpoint-200.pt \
    --num-points 61 \
    --dataset CIFAR10 \
    --data-path ./data \
    --model VGG16 \
    --transform VGG \
    --batch-size 128 \
    --num-workers 4 \
    --use-test

if [ $? -ne 0 ]; then
    echo "Linear interpolation evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "RANDOM PERMUTATION CREATION AND EVALUATION COMPLETED"
echo "========================================"
