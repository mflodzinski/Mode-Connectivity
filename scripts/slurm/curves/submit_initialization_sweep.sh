#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_init_sweep_%j.out
#SBATCH --error=slurm_init_sweep_%j.err
#SBATCH --job-name=init_sweep
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "================================================================================"
echo "INITIALIZATION SWEEP: TRAINING BEZIER CURVES WITH DIFFERENT INITIALIZATIONS"
echo "================================================================================"
echo ""
echo "Training curves with same endpoints but different initialization methods:"
echo "  - Biased linear (5 variants: alpha = 0.1, 0.25, 0.5, 0.75, 0.9)"
echo "  - Perturbed linear (3 variants: noise = 0.01, 0.05, 0.1)"
echo "  - Sphere-constrained (2 variants: inside and outside)"
echo ""
echo "All experiments use:"
echo "  - Endpoints: seed0 and seed1 (checkpoint-200.pt)"
echo "  - Epochs: 100"
echo "  - Learning rate: 0.015"
echo "  - Weight decay: 0.0 (no regularization)"
echo "  - Bezier curve with 3 bends"
echo "================================================================================"
echo ""

# Array of config files
CONFIGS=(
    "vgg16_curve_init_alpha0.75"
    "vgg16_curve_init_alpha0.9"
    "vgg16_curve_init_perturbed_small"
    "vgg16_curve_init_perturbed_large"
    "vgg16_curve_init_sphere_inside"
    "vgg16_curve_init_sphere_outside"
)

# Train each configuration
for CONFIG in "${CONFIGS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "Training: ${CONFIG}"
    echo "--------------------------------------------------------------------------------"

    # Extract output_root from config file to get correct checkpoint directory
    CONFIG_FILE="configs/garipov/curves_init/${CONFIG}.yaml"
    OUTPUT_ROOT=$(grep "^output_root:" "${CONFIG_FILE}" | sed 's/output_root: *"\(.*\)"/\1/')

    # Create output directory if it doesn't exist
    echo "Creating output directory: ${OUTPUT_ROOT}"
    mkdir -p "${OUTPUT_ROOT}"

    # Run training
    echo "Starting training..."
    srun python scripts/train/run_garipov_curve.py \
        --config-name="${CONFIG}"

    if [ $? -eq 0 ]; then
        echo "✓ ${CONFIG} training complete"

        # Show checkpoint info
        CHECKPOINT="${OUTPUT_ROOT}/checkpoint-100.pt"
        if [ -f "${CHECKPOINT}" ]; then
            echo "  Final checkpoint: ${CHECKPOINT}"
            ls -lh "${CHECKPOINT}"
        else
            echo "  Warning: Expected checkpoint not found at ${CHECKPOINT}"
        fi
    else
        echo "✗ ${CONFIG} training failed"
    fi

    echo ""
done

echo "================================================================================"
echo "INITIALIZATION SWEEP COMPLETE"
echo "================================================================================"
echo ""
echo "Summary of trained curves:"
for CONFIG in "${CONFIGS[@]}"; do
    # Extract output_root from config file
    CONFIG_FILE="configs/garipov/curves_init/${CONFIG}.yaml"
    OUTPUT_ROOT=$(grep "^output_root:" "${CONFIG_FILE}" | sed 's/output_root: *"\(.*\)"/\1/')
    CHECKPOINT="${OUTPUT_ROOT}/checkpoint-100.pt"

    if [ -f "${CHECKPOINT}" ]; then
        SIZE=$(ls -lh "${CHECKPOINT}" | awk '{print $5}')
        echo "✓ ${CONFIG}: ${SIZE}"
    else
        echo "✗ ${CONFIG}: MISSING"
    fi
done
echo ""
echo "================================================================================"
