#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=0:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err
#SBATCH --job-name=verify_lmc
#SBATCH --gres=gpu:a40:1

# Verify LMC connectivity between w_0 and w_1 from shared init training
# Evaluates linear interpolation barrier to confirm low barrier (<1%)

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
PROJECT_ROOT="/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity"
export PYTHONPATH=${PROJECT_ROOT}:${PROJECT_ROOT}/scripts:$PYTHONPATH

# Checkpoints
ENDPOINT0="results/vgg16/cifar10/endpoints/lmc_connected/seed0/checkpoint-200.pt"
ENDPOINT1="results/vgg16/cifar10/endpoints/lmc_connected/seed1/checkpoint-200.pt"
OUTPUT_DIR="results/vgg16/cifar10/endpoints/lmc_connected/evaluations"

mkdir -p "${OUTPUT_DIR}"

# Step 1: Calculate L2 distance between the two modes
echo "========================================"
echo "STEP 1: Computing L2 distance between modes"
echo "========================================"
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

echo ""
# Step 2: Evaluate linear interpolation using evaluate.py
echo "========================================"
echo "STEP 2: Evaluating linear interpolation"
echo "========================================"
srun python scripts/eval/evaluate.py \
    --mode linear \
    --init-start "${ENDPOINT0}" \
    --init-end "${ENDPOINT1}" \
    --num-points 61 \
    --dataset CIFAR10 \
    --model VGG16 \
    --data-path data \
    --dir "${OUTPUT_DIR}"

echo ""
echo "========================================"
echo "EVALUATION COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/linear.npz"
echo ""
echo "To download:"
echo "  scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/$OUTPUT_DIR/linear.npz ."
