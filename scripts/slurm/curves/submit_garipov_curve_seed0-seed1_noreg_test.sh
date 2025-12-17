#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_curve_init_test_%j.out
#SBATCH --error=slurm_curve_init_test_%j.err
#SBATCH --job-name=curve_init_test
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "================================================================================"
echo "TEST RUN: CURVE TRAINING WITH PERTURBED INITIALIZATION (small noise)"
echo "================================================================================"
echo ""
echo "This test will:"
echo "  1. Initialize middle point with perturbed linear (alpha=0.5, noise=0.01)"
echo "  2. Train curve for 10 epochs (no regularization)"
echo "  3. Verify L2 norms are calculated at epoch 0"
echo "  4. Evaluate the trained curve (61 points)"
echo ""
echo "Config: vgg16_curve_seed0-seed1_noreg_test"
echo "Initialization: perturbed (alpha=0.5, noise=0.01)"
echo ""
echo "--------------------------------------------------------------------------------"

# Create output directories
mkdir -p results/vgg16/cifar10/curves/initialization/test_perturbed_small/checkpoints
mkdir -p results/vgg16/cifar10/curves/initialization/test_perturbed_small/evaluations

echo ""
echo "STEP 1: Training curve for 10 epochs"
echo "--------------------------------------------------------------------------------"

# Run curve training
srun python scripts/train/run_garipov_curve.py --config-name vgg16_curve_seed0-seed1_noreg_test

if [ $? -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Training failed"
    echo "================================================================================"
    exit 1
fi

echo ""
echo "✓ Training complete"
echo ""
echo "STEP 2: Evaluating trained curve"
echo "--------------------------------------------------------------------------------"

CHECKPOINT="results/vgg16/cifar10/curves/initialization/test_perturbed_small/checkpoints/checkpoint-10.pt"
EVAL_DIR="results/vgg16/cifar10/curves/initialization/test_perturbed_small/evaluations"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "⚠️  ERROR: Checkpoint not found: ${CHECKPOINT}"
    echo "Training may have failed to save checkpoint"
    exit 1
fi

echo "Checkpoint: ${CHECKPOINT}"
echo "Output dir: ${EVAL_DIR}"
echo ""

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
echo "================================================================================"
echo "TEST RUN SUMMARY"
echo "================================================================================"
echo ""
echo "Training outputs:"
ls -lh results/vgg16/cifar10/curves/initialization/test_perturbed_small/checkpoints/
echo ""
echo "Evaluation outputs:"
ls -lh results/vgg16/cifar10/curves/initialization/test_perturbed_small/evaluations/
echo ""
echo "L2 norm files:"
ls -lh results/vgg16/cifar10/curves/initialization/test_perturbed_small/evaluations/*l2*.npz 2>/dev/null || echo "No L2 norm files found"
echo ""
echo "================================================================================"
echo "VERIFICATION CHECKLIST"
echo "================================================================================"
echo ""
echo "To verify the corrected training script works correctly:"
echo ""
echo "1. Check training log for epoch 0 L2 norms:"
echo "   grep 'Initial middle point L2 norm' slurm_curve_init_test_*.out"
echo "   grep 'Initial interpolated L2 norm' slurm_curve_init_test_*.out"
echo ""
echo "2. Check saved L2 norm file contains epoch 0:"
echo "   python -c \"import numpy as np; data=np.load('${EVAL_DIR}/middle_point_l2_norms.npz'); print('Epochs:', data['epochs']); print('First epoch:', data['epochs'][0])\""
echo ""
echo "3. Expected results:"
echo "   - Initial middle point L2 norm should be ~27-30 (perturbed small noise)"
echo "   - Epochs array should start with 0"
echo "   - L2 norms should be tracked for epochs 0-10"
echo ""
echo "================================================================================"
echo "TEST RUN COMPLETE"
echo "================================================================================"
