#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_align_seed0_seed1_%j.out
#SBATCH --error=slurm_align_seed0_seed1_%j.err
#SBATCH --job-name=align_seed0_seed1
#SBATCH --gres=gpu:a40:1

# Align independently trained seed0 and seed1 models using weight matching
# Tests if alignment can reduce the barrier between models with different initializations

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directory
mkdir -p results/analysis/alignment_independent

# Checkpoints
SEED0="results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt"
SEED1="results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt"

echo "========================================"
echo "Aligning independently trained models"
echo "========================================"
echo "Model A (reference): $SEED0"
echo "Model B (to align):  $SEED1"
echo ""

# Run alignment
srun python scripts/analysis/align_independent_models.py \
    --model-a $SEED0 \
    --model-b $SEED1 \
    --method weight_matching \
    --max-iter 100 \
    --num-eval-points 61 \
    --data-path ./data \
    --output results/analysis/alignment_independent/seed0_seed1_results.json

echo ""
echo "========================================"
echo "ALIGNMENT COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: results/analysis/alignment_independent/seed0_seed1_results.json"
echo ""
echo "To download:"
echo "  scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/analysis/alignment_independent/seed0_seed1_results.json ."
