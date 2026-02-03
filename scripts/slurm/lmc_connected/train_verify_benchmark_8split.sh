#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_8split_full_%j.out
#SBATCH --error=slurm_lmc_8split_full_%j.err
#SBATCH --job-name=lmc_8split_full
#SBATCH --gres=gpu:a40:1

# Complete pipeline for 8/192 split:
# 1. Train LMC-connected pair (8 epochs shared, 192 epochs split)
# 2. Verify LMC connectivity (L2 distance + linear interpolation)
# 3. Run permutation alignment benchmark

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:$PYTHONPATH

# Paths
OUTPUT_DIR="results/vgg16/cifar10/endpoints/lmc_connected_8split"
ENDPOINT0="${OUTPUT_DIR}/seed0/checkpoint-200.pt"
ENDPOINT1="${OUTPUT_DIR}/seed1/checkpoint-200.pt"
EVAL_DIR="${OUTPUT_DIR}/evaluations"
BENCHMARK_DIR="results/analysis/alignment_benchmark_8split"

# Create output directories
mkdir -p "${EVAL_DIR}"
mkdir -p "${BENCHMARK_DIR}"

# ============================================================================
# PHASE 1: TRAINING
# ============================================================================
echo "========================================"
echo "PHASE 1: Training LMC-connected pair (8/192 split)"
echo "========================================"
echo "Stage 1: Train from scratch for 8 epochs"
echo "Stage 2: Split with different batch seeds (8 -> 200)"
echo "Output: ${OUTPUT_DIR}/"
echo ""

srun python scripts/train/run_lmc_connected_pair_from_scratch.py \
    --config-name vgg16_lmc_connected_pair_8split

echo ""
echo "========================================"
echo "TRAINING COMPLETE!"
echo "========================================"
echo ""
echo "Checkpoints saved to:"
echo "  - ${OUTPUT_DIR}/shared/checkpoint-8.pt"
echo "  - ${ENDPOINT0}"
echo "  - ${ENDPOINT1}"
echo ""

# ============================================================================
# PHASE 2: VERIFICATION
# ============================================================================
echo "========================================"
echo "PHASE 2: Verifying LMC connectivity"
echo "========================================"

# Step 2a: Calculate L2 distance between the two modes
echo ""
echo "----------------------------------------"
echo "Step 2a: Computing L2 distance between modes"
echo "----------------------------------------"
srun python -c "
import torch

ckpt0 = torch.load('${ENDPOINT0}', map_location='cpu')
ckpt1 = torch.load('${ENDPOINT1}', map_location='cpu')

state0 = ckpt0['model_state']
state1 = ckpt1['model_state']

# Compute L2 distance
total_diff_sq = 0.0
total_params = 0
for key in state0:
    diff = state0[key].float() - state1[key].float()
    total_diff_sq += (diff ** 2).sum().item()
    total_params += diff.numel()

l2_dist = total_diff_sq ** 0.5
print(f'L2 distance between w_0 and w_1: {l2_dist:.4f}')
print(f'Total parameters: {total_params:,}')
print(f'RMS difference: {(total_diff_sq / total_params) ** 0.5:.6f}')
"

# Step 2b: Evaluate linear interpolation
echo ""
echo "----------------------------------------"
echo "Step 2b: Evaluating linear interpolation"
echo "----------------------------------------"
srun python scripts/eval/evaluate.py \
    --mode linear \
    --init-start "${ENDPOINT0}" \
    --init-end "${ENDPOINT1}" \
    --num-points 61 \
    --dataset CIFAR10 \
    --model VGG16 \
    --data-path data \
    --dir "${EVAL_DIR}"

echo ""
echo "========================================"
echo "VERIFICATION COMPLETE!"
echo "========================================"
echo "Linear interpolation results: ${EVAL_DIR}/linear.npz"
echo ""

# ============================================================================
# PHASE 3: PERMUTATION ALIGNMENT BENCHMARK
# ============================================================================
echo "========================================"
echo "PHASE 3: Permutation Alignment Benchmark"
echo "========================================"
echo "w_0: ${ENDPOINT0}"
echo "w_1: ${ENDPOINT1}"
echo ""

srun python scripts/analysis/benchmark_alignment.py \
    --w0 "${ENDPOINT0}" \
    --w1 "${ENDPOINT1}" \
    --perm-seed 42 \
    --method weight_matching \
    --max-iter 100 \
    --num-eval-points 61 \
    --data-path ./data \
    --output "${BENCHMARK_DIR}/results.json"

echo ""
echo "========================================"
echo "ALL PHASES COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  - Checkpoints: ${OUTPUT_DIR}/"
echo "  - Linear interpolation: ${EVAL_DIR}/linear.npz"
echo "  - Benchmark results: ${BENCHMARK_DIR}/results.json"
echo ""
echo "To download results:"
echo "  mkdir -p ./results/vgg16/cifar10/endpoints/lmc_connected_8split/evaluations ./results/analysis/alignment_benchmark_8split"
echo "  scp mlodzinski@login.daic.tudelft.nl:${PROJECT_ROOT}/${EVAL_DIR}/linear.npz ./results/vgg16/cifar10/endpoints/lmc_connected_8split/evaluations/"
echo "  scp mlodzinski@login.daic.tudelft.nl:${PROJECT_ROOT}/${BENCHMARK_DIR}/results.json ./results/analysis/alignment_benchmark_8split/"
echo "  scp mlodzinski@login.daic.tudelft.nl:${PROJECT_ROOT}/${ENDPOINT0} ./results/vgg16/cifar10/endpoints/lmc_connected_8split/seed0/"
echo "  scp mlodzinski@login.daic.tudelft.nl:${PROJECT_ROOT}/${ENDPOINT1} ./results/vgg16/cifar10/endpoints/lmc_connected_8split/seed1/"
