#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_reevaluate_init_curves_%j.out
#SBATCH --error=slurm_reevaluate_init_curves_%j.err
#SBATCH --job-name=reevaluate_init_curves
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "================================================================================"
echo "RE-EVALUATING ALL INITIALIZATION CURVE EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Purpose: Fix incorrect curve.npz files that don't match trained checkpoints"
echo ""
echo "Issue found: Existing curve.npz files show wrong L2 norms along the path."
echo "  - Training logs show correct interpolated L2 values"
echo "  - curve.npz files were generated with outdated/incorrect checkpoints"
echo ""
echo "This script will:"
echo "  1. Back up existing curve.npz files"
echo "  2. Re-evaluate each checkpoint-150.pt with eval_curve.py"
echo "  3. Generate new, correct curve.npz files"
echo ""
echo "================================================================================"
echo ""

# Array of experiments to re-evaluate
declare -A EXPERIMENTS
EXPERIMENTS["biased_linear_alpha0.9"]="results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.9"
EXPERIMENTS["perturbed_noise0.01"]="results/vgg16/cifar10/curves/initialization/perturbed/noise_0.01"
EXPERIMENTS["perturbed_noise0.1"]="results/vgg16/cifar10/curves/initialization/perturbed/noise_0.1"
EXPERIMENTS["sphere_inside"]="results/vgg16/cifar10/curves/initialization/sphere_constrained/inside"
EXPERIMENTS["sphere_outside"]="results/vgg16/cifar10/curves/initialization/sphere_constrained/outside"

for NAME in "${!EXPERIMENTS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "Re-evaluating: ${NAME}"
    echo "--------------------------------------------------------------------------------"

    BASE_DIR="${EXPERIMENTS[$NAME]}"
    CHECKPOINT="${BASE_DIR}/checkpoints/checkpoint-150.pt"
    EVAL_DIR="${BASE_DIR}/evaluations"

    # Check if checkpoint exists
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "⚠️  WARNING: Checkpoint not found: ${CHECKPOINT}"
        echo "Skipping ${NAME}"
        echo ""
        continue
    fi

    # Run evaluation
    echo "Running eval_curve.py..."
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

    if [ $? -eq 0 ]; then
        echo "✓ ${NAME} re-evaluation complete"

        # Verify the new file was created
        if [ -f "${EVAL_DIR}/curve.npz" ]; then
            echo "  New curve.npz created successfully"

            # Extract L2 at t=0.5 for verification
            python -c "
import numpy as np
data = np.load('${EVAL_DIR}/curve.npz')
l2_mid = data['l2_norm'][30]  # t=0.5 is at index 30 for 61 points
print(f'  L2 at t=0.5: {l2_mid:.6f}')

# Compare with training logs
mp_data = np.load('${EVAL_DIR}/middle_point_l2_norms.npz')
training_l2 = mp_data['interpolated_l2_norms'][-1]
print(f'  Training log L2: {training_l2:.6f}')
diff = abs(l2_mid - training_l2)
print(f'  Difference: {diff:.6f}')

if diff < 0.01:
    print('  ✓ Values match!')
else:
    print(f'  ⚠️  Warning: Discrepancy of {diff:.6f}')
" || echo "  (Could not verify L2 values)"
        else
            echo "  ⚠️  WARNING: curve.npz was not created"
        fi
    else
        echo "✗ ${NAME} re-evaluation failed"
    fi

    echo ""
done

echo "================================================================================"
echo "RE-EVALUATION COMPLETE"
echo "================================================================================"
echo ""
echo "Summary of updated files:"
for NAME in "${!EXPERIMENTS[@]}"; do
    BASE_DIR="${EXPERIMENTS[$NAME]}"
    EVAL_DIR="${BASE_DIR}/evaluations"

    if [ -f "${EVAL_DIR}/curve.npz" ]; then
        echo "${NAME}:"
        echo "  curve.npz: $(ls -lh "${EVAL_DIR}/curve.npz" | awk '{print $9 " (" $5 ")"}')"

        # Show backup if exists
        LATEST_BACKUP=$(ls -t "${EVAL_DIR}"/curve.npz.backup_* 2>/dev/null | head -1)
        if [ -n "${LATEST_BACKUP}" ]; then
            echo "  backup: $(basename "${LATEST_BACKUP}")"
        fi
    else
        echo "${NAME}: No curve.npz file"
    fi
done

echo ""
echo "================================================================================"
echo "NEXT STEPS"
echo "================================================================================"
echo ""
echo "After copying results to local machine:"
echo "  1. Re-generate connectivity plots with correct curve.npz files"
echo "  2. Verify that plot 6 (L2 evolution) now aligns with plot 5 (L2 along path)"
echo ""
echo "================================================================================"
