#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_benchmark_80split_%j.out
#SBATCH --error=slurm_benchmark_80split_%j.err
#SBATCH --job-name=benchmark_80split
#SBATCH --gres=gpu:a40:1

# Benchmark permutation alignment on 80/120 split LMC-connected pair
# Tests if weight matching can recover known permutations

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directory
mkdir -p results/analysis/alignment_benchmark_80split

# Checkpoints from 80/120 split training
W0="results/vgg16/cifar10/endpoints/lmc_connected_80split/seed0/checkpoint-200.pt"
W1="results/vgg16/cifar10/endpoints/lmc_connected_80split/seed1/checkpoint-200.pt"

echo "========================================"
echo "Permutation Alignment Benchmark (80/120 split)"
echo "========================================"
echo "w_0: $W0"
echo "w_1: $W1"
echo ""

# Run benchmark
srun python scripts/analysis/benchmark_alignment.py \
    --w0 $W0 \
    --w1 $W1 \
    --perm-seed 42 \
    --method weight_matching \
    --max-iter 100 \
    --num-eval-points 61 \
    --data-path ./data \
    --output results/analysis/alignment_benchmark_80split/results.json

echo ""
echo "========================================"
echo "BENCHMARK COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: results/analysis/alignment_benchmark_80split/results.json"
echo ""
echo "To download:"
echo "  scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/analysis/alignment_benchmark_80split/results.json ."
