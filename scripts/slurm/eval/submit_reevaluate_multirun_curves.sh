#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_reevaluate_multirun_%j.out
#SBATCH --error=slurm_reevaluate_multirun_%j.err
#SBATCH --job-name=reevaluate_multirun
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "================================================================================"
echo "RE-EVALUATING MULTIRUN CURVE EXPERIMENTS"
echo "================================================================================"
echo ""
echo "Purpose: Generate correct curve.npz files for multirun experiments"
echo ""
echo "This will re-evaluate checkpoint-200.pt for each multirun experiment:"
echo "  - seed0-seed1_noreg/run_seed0"
echo "  - seed0-seed1_noreg/run_seed42"
echo "  - seed0-seed1_noreg/run_seed123"
echo ""
echo "================================================================================"
echo ""

# Array of multirun experiments to re-evaluate
declare -A EXPERIMENTS
EXPERIMENTS["run_seed0"]="results/vgg16/cifar10/curves/multirun/seed0-seed1_noreg/run_seed0"
EXPERIMENTS["run_seed42"]="results/vgg16/cifar10/curves/multirun/seed0-seed1_noreg/run_seed42"
EXPERIMENTS["run_seed123"]="results/vgg16/cifar10/curves/multirun/seed0-seed1_noreg/run_seed123"

for NAME in "${!EXPERIMENTS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "Re-evaluating: ${NAME}"
    echo "--------------------------------------------------------------------------------"

    BASE_DIR="${EXPERIMENTS[$NAME]}"
    CHECKPOINT="${BASE_DIR}/checkpoints/checkpoint-200.pt"
    EVAL_DIR="${BASE_DIR}/evaluations"

    # Check if checkpoint exists
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "⚠️  WARNING: Checkpoint not found: ${CHECKPOINT}"
        echo "Skipping ${NAME}"
        echo ""
        continue
    fi

    echo "Checkpoint: ${CHECKPOINT}"
    echo "Output directory: ${EVAL_DIR}"
    echo ""

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

# Compare with training logs if available
try:
    mp_data = np.load('${EVAL_DIR}/middle_point_l2_norms.npz')
    training_l2 = mp_data['interpolated_l2_norms'][-1]
    print(f'  Training log L2: {training_l2:.6f}')
    diff = abs(l2_mid - training_l2)
    print(f'  Difference: {diff:.6f}')
    if diff < 0.01:
        print('  ✓ Values match!')
    else:
        print(f'  ⚠️  Warning: Discrepancy of {diff:.6f}')
except:
    print('  (No training logs found for comparison)')
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
    else
        echo "${NAME}: No curve.npz file"
    fi
done

echo ""
echo "================================================================================"
echo "NEXT STEPS"
echo "================================================================================"
echo ""
echo "Copy results to local machine with:"
echo ""
echo "scp cluster:.../multirun/seed0-seed1_noreg/run_seed0/evaluations/curve.npz ..."
echo "scp cluster:.../multirun/seed0-seed1_noreg/run_seed42/evaluations/curve.npz ..."
echo "scp cluster:.../multirun/seed0-seed1_noreg/run_seed123/evaluations/curve.npz ..."
echo ""
echo "================================================================================"
