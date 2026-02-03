#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_lmc_80split_%j.out
#SBATCH --error=slurm_lmc_80split_%j.err
#SBATCH --job-name=lmc_80split
#SBATCH --gres=gpu:a40:1

# Train LMC-connected pair from scratch with 80 epochs shared, 120 epochs split
# Stage 1: Train from scratch for 80 epochs -> shared checkpoint
# Stage 2: Split training with different batch seeds (80 -> 200)

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "========================================"
echo "Training LMC-connected pair (80/120 split)"
echo "========================================"
echo "Stage 1: Train from scratch for 80 epochs"
echo "Stage 2: Split with different batch seeds (80 -> 200)"
echo "Output: results/vgg16/cifar10/endpoints/lmc_connected_80split/"
echo ""

# Run the training script (trains from scratch)
srun python scripts/train/run_lmc_connected_pair_from_scratch.py \
    --config-name vgg16_lmc_connected_pair_80split

echo ""
echo "========================================"
echo "TRAINING COMPLETE!"
echo "========================================"
echo ""
echo "Checkpoints saved to:"
echo "  - results/vgg16/cifar10/endpoints/lmc_connected_80split/shared/checkpoint-80.pt"
echo "  - results/vgg16/cifar10/endpoints/lmc_connected_80split/seed0/checkpoint-200.pt"
echo "  - results/vgg16/cifar10/endpoints/lmc_connected_80split/seed1/checkpoint-200.pt"
echo ""
echo "To verify LMC connectivity, run:"
echo "  sbatch scripts/slurm/lmc_connected/verify_lmc_connectivity_80split.sh"
