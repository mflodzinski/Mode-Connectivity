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

# Array of config names and their checkpoint paths
declare -A CONFIGS
CONFIGS["alpha0.75"]="results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.75/checkpoints"
CONFIGS["alpha0.9"]="results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.9/checkpoints"
CONFIGS["perturbed_small"]="results/vgg16/cifar10/curves/initialization/perturbed_linear/noise_0.01/checkpoints"
CONFIGS["perturbed_large"]="results/vgg16/cifar10/curves/initialization/perturbed_linear/noise_0.1/checkpoints"
CONFIGS["sphere_inside"]="results/vgg16/cifar10/curves/initialization/sphere_constrained/inside/checkpoints"
CONFIGS["sphere_outside"]="results/vgg16/cifar10/curves/initialization/sphere_constrained/outside/checkpoints"

for NAME in "${!CONFIGS[@]}"; do
    echo "--------------------------------------------------------------------------------"
    echo "Evaluating: ${NAME}"
    echo "--------------------------------------------------------------------------------"

    CKPT_DIR="${CONFIGS[$NAME]}"
    CHECKPOINT="${CKPT_DIR}/checkpoint-100.pt"

    # Check if checkpoint exists
    if [ ! -f "${CHECKPOINT}" ]; then
        echo "⚠️  WARNING: Checkpoint not found: ${CHECKPOINT}"
        echo "Skipping ${NAME}"
        echo ""
        continue
    fi

    # Create evaluations directory (same parent as checkpoints)
    EVAL_DIR="${CKPT_DIR%/checkpoints}/evaluations"
    mkdir -p "${EVAL_DIR}"

    echo "Checkpoint: ${CHECKPOINT}"
    echo "Output dir: ${EVAL_DIR}"

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
        echo "✓ ${NAME} evaluation complete"
        echo "  Results saved to: ${EVAL_DIR}"
        ls -lh "${EVAL_DIR}"/*.npz 2>/dev/null || echo "  No .npz files found"
    else
        echo "✗ ${NAME} evaluation failed"
    fi

    echo ""
done

echo "================================================================================"
echo "ALL EVALUATIONS COMPLETE"
echo "================================================================================"
echo ""
echo "Summary of generated files:"
for NAME in "${!CONFIGS[@]}"; do
    CKPT_DIR="${CONFIGS[$NAME]}"
    EVAL_DIR="${CKPT_DIR%/checkpoints}/evaluations"

    if [ -d "${EVAL_DIR}" ]; then
        echo "${NAME}:"
        ls -lh "${EVAL_DIR}"/*.npz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}' || echo "  No files"
    fi
done
echo ""
echo "================================================================================"
