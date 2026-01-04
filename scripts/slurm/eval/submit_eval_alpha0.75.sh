#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_alpha0.75_%j.out
#SBATCH --error=slurm_eval_alpha0.75_%j.err
#SBATCH --job-name=eval_alpha0.75
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "================================================================================"
echo "RE-EVALUATING BIASED LINEAR ALPHA=0.75 CURVE"
echo "================================================================================"
echo ""
echo "Purpose: Generate correct curve.npz file for biased_linear alpha_0.75"
echo ""
echo "Issue: Existing curve.npz shows L2=32.59 at t=0.5"
echo "       Training logs show L2=35.11 at t=0.5"
echo "       Checkpoint contains middle point that should give L2=35.11"
echo ""
echo "This will:"
echo "  2. Re-evaluate checkpoint-150.pt"
echo "  3. Verify new curve.npz matches training logs"
echo ""
echo "================================================================================"
echo ""

CHECKPOINT="results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.75/checkpoints/checkpoint-150.pt"
EVAL_DIR="results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.75/evaluations"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "⚠️  ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "Output directory: ${EVAL_DIR}"
echo ""


echo "Running eval_curve.py..."
echo "--------------------------------------------------------------------------------"

# Run evaluation
srun python external/dnn-mode-connectivity/eval_curve.py \
  --dir "${EVAL_DIR}" \
  --dataset CIFAR10 \
  --data_path ./data \
  --transform VGG \
  --model VGG16 \
  --curve Bezier \
  --num_bends 3 \
  --ckpt "${CHECKPOINT}" \
  --num_points 61 \
  --use_test

if [ $? -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Evaluation failed"
    echo "================================================================================"
    exit 1
fi

echo ""
echo "✓ Evaluation complete"
echo ""

# Verify the new file
if [ -f "${EVAL_DIR}/curve.npz" ]; then
    echo "================================================================================"
    echo "VERIFICATION"
    echo "================================================================================"
    echo ""

    python -c "
import numpy as np

print('New curve.npz file:')
curve_data = np.load('${EVAL_DIR}/curve.npz')
l2_t0 = curve_data['l2_norm'][0]
l2_t05 = curve_data['l2_norm'][30]
l2_t1 = curve_data['l2_norm'][60]

print(f'  L2 at t=0.0: {l2_t0:.6f}')
print(f'  L2 at t=0.5: {l2_t05:.6f}')
print(f'  L2 at t=1.0: {l2_t1:.6f}')
print()

print('Training logs (middle_point_l2_norms.npz):')
mp_data = np.load('${EVAL_DIR}/middle_point_l2_norms.npz')
training_l2 = mp_data['interpolated_l2_norms'][-1]
print(f'  Final interpolated L2: {training_l2:.6f}')
print()

diff = abs(l2_t05 - training_l2)
print(f'Difference: {diff:.6f}')
print()

if diff < 0.01:
    print('✓ SUCCESS: curve.npz matches training logs!')
else:
    print(f'⚠️  WARNING: Discrepancy of {diff:.6f}')
    print('   Expected: ~35.11')
    print(f'   Got:      {l2_t05:.6f}')
"

    echo ""
    echo "================================================================================"
    echo "FILES"
    echo "================================================================================"
    echo ""
    ls -lh "${EVAL_DIR}"/*.npz
    echo ""
else
    echo "⚠️  ERROR: curve.npz was not created"
    exit 1
fi

echo "================================================================================"
echo "COMPLETE"
echo "================================================================================"
echo ""
echo "Next steps:"
echo "  1. Copy new curve.npz to local machine:"
echo "     scp cluster:${EVAL_DIR}/curve.npz results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.75/evaluations/"
echo ""
echo "  2. Re-generate connectivity plots locally"
echo ""
echo "================================================================================"
