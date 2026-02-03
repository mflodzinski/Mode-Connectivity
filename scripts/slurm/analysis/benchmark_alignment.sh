#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --job-name=benchmark_alignment
#SBATCH --gres=gpu:a40:1

# Benchmark permutation alignment methods
# Tests if weight matching can recover known permutations between LMC-connected modes

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directory
mkdir -p results/analysis/alignment_benchmark

# Run benchmark
srun python scripts/analysis/benchmark_alignment.py \
    --w0 results/vgg16/cifar10/endpoints/lmc_connected/seed0/checkpoint-200.pt \
    --w1 results/vgg16/cifar10/endpoints/lmc_connected/seed1/checkpoint-200.pt \
    --perm-seed 42 \
    --method weight_matching \
    --max-iter 100 \
    --num-eval-points 11 \
    --data-path ./data \
    --output results/analysis/alignment_benchmark/results.json
