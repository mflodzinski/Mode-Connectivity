#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_init_sweep_%j.out
#SBATCH --error=slurm_eval_init_sweep_%j.err
#SBATCH --job-name=eval_init_sweep
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo "================================================================================"
echo "EVALUATING ALL INITIALIZATION EXPERIMENTS"
echo "================================================================================"
echo ""

# Array of experiment directories
EXPERIMENTS=(
    "curve_init_alpha0.1"
    "curve_init_alpha0.25"
    "curve_init_alpha0.5"
    "curve_init_alpha0.75"
    "curve_init_alpha0.9"
    "curve_init_perturbed_small"
    "curve_init_perturbed_medium"
    "curve_init_perturbed_large"
    "curve_init_sphere_inside"
    "curve_init_sphere_outside"
)

for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "Evaluating: ${EXPERIMENT}"
    echo "--------------------------------------------------------------------------------"

    # Check if checkpoint exists
    CHECKPOINT="results/vgg16/cifar10/${EXPERIMENT}/checkpoints/checkpoint-200.pt"
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "⚠️  WARNING: Checkpoint not found: ${CHECKPOINT}"
        echo "Skipping ${EXPERIMENT}"
        echo ""
        continue
    fi

    # Create evaluations directory
    EVAL_DIR="results/vgg16/cifar10/${EXPERIMENT}/evaluations"
    mkdir -p "${EVAL_DIR}"

    # Run evaluation
    echo "Running evaluation..."
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
        echo "✓ ${EXPERIMENT} evaluation complete"
        echo "  Results saved to: ${EVAL_DIR}"
        ls -lh "${EVAL_DIR}"/*.npz
    else
        echo "✗ ${EXPERIMENT} evaluation failed"
    fi

    echo ""
done

echo "================================================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "================================================================================"
echo ""
echo "Summary of generated files:"
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    EVAL_DIR="results/vgg16/cifar10/${EXPERIMENT}/evaluations"

    if [ -d "${EVAL_DIR}" ]; then
        echo "${EXPERIMENT}:"
        ls -lh "${EVAL_DIR}"/*.npz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    fi
done
echo ""
echo "================================================================================"
