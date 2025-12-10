#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=4G
#SBATCH --job-name=eval_multirun_all
#SBATCH --error=slurm_eval_polygon_seed0-mirror_%j.err
#SBATCH --output=slurm_eval_polygon_seed0-mirror_%j.out

export PYTHONPATH="${PYTHONPATH}:${PWD}"

echo "================================================================================"
echo "EVALUATING ALL MULTIRUN EXPERIMENTS"
echo "================================================================================"
echo ""

# Array of experiment suffixes
SEEDS=("seed0" "seed42" "seed123")

for SEED in "${SEEDS[@]}"; do
    EXPERIMENT="curve_seed0-seed1_noreg_multirun_${SEED}"

    echo "--------------------------------------------------------------------------------"
    echo "Evaluating: ${EXPERIMENT}"
    echo "--------------------------------------------------------------------------------"

    # Check if checkpoint exists
    CHECKPOINT="results/vgg16/cifar10/${EXPERIMENT}/checkpoints/checkpoint-50.pt"
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
for SEED in "${SEEDS[@]}"; do
    EXPERIMENT="curve_seed0-seed1_noreg_multirun_${SEED}"
    EVAL_DIR="results/vgg16/cifar10/${EXPERIMENT}/evaluations"

    if [ -d "${EVAL_DIR}" ]; then
        echo "${EXPERIMENT}:"
        ls -lh "${EVAL_DIR}"/*.npz 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    fi
done
echo ""
echo "================================================================================"
