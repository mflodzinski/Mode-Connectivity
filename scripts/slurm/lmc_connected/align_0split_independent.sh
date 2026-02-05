#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_align_0split_%j.out
#SBATCH --error=slurm_align_0split_%j.err
#SBATCH --job-name=align_0split
#SBATCH --gres=gpu:a40:1

# Permutation alignment between 0/200 split endpoints (same random init, different seeds)
# This is similar to seed0-seed1 experiment but with shared random initialization
#
# Key difference from benchmark_alignment.py:
# - No w1' construction (no random permutation applied)
# - Direct alignment between two models trained from same init
# - Tests if weight matching can align models that diverged during training

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:$PYTHONPATH

# Paths
MODEL_A="results/vgg16/cifar10/endpoints/lmc_connected_0split/seed0/checkpoint-200.pt"
MODEL_B="results/vgg16/cifar10/endpoints/lmc_connected_0split/seed1/checkpoint-200.pt"
OUTPUT_DIR="results/analysis/alignment_0split_independent"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# ============================================================================
# Run alignment between 0/200 split endpoints
# ============================================================================
echo "========================================"
echo "Permutation Alignment: 0/200 Split"
echo "========================================"
echo ""
echo "Model A (reference): ${MODEL_A}"
echo "Model B (to align):  ${MODEL_B}"
echo ""
echo "This experiment:"
echo "  - Both models trained from SAME random init (seed=42)"
echo "  - Different training seeds (0 and 1)"
echo "  - No shared training epochs (0/200 split)"
echo "  - Tests weight matching alignment quality"
echo ""

srun python scripts/analysis/align_independent_models.py \
    --model-a "${MODEL_A}" \
    --model-b "${MODEL_B}" \
    --method weight_matching \
    --max-iter 100 \
    --num-eval-points 61 \
    --data-path ./data \
    --batch-size 128 \
    --output "${OUTPUT_DIR}/results.json"

echo ""
echo "========================================"
echo "ALIGNMENT COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: ${OUTPUT_DIR}/results.json"
echo ""
echo "To download results:"
echo "  mkdir -p ./results/analysis/alignment_0split_independent"
echo "  scp mlodzinski@login.daic.tudelft.nl:${PROJECT_ROOT}/${OUTPUT_DIR}/results.json ./results/analysis/alignment_0split_independent/"
