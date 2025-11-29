#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_pred_changes_eval_%j.out
#SBATCH --error=slurm_pred_changes_eval_%j.err
#SBATCH --job-name=pred_changes_eval

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

echo ""
echo "========================================"
echo "PREDICTION CHANGES EVALUATION (STEP 1)"
echo "========================================"
echo ""
echo "This script collects detailed predictions and features"
echo "at 61 points along the Bezier curve for prediction change analysis."
echo ""

# Check if checkpoint exists
CHECKPOINT="results/vgg16/cifar10/curve/checkpoints/checkpoint-200.pt"
if [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo "Found checkpoint: $CHECKPOINT"
echo ""

# Run detailed evaluation
echo "Starting detailed evaluation..."
echo ""

srun python scripts/eval/eval_curve_detailed.py \
    --curve_ckpt results/vgg16/cifar10/curve/checkpoints/checkpoint-200.pt \
    --endpoint0_ckpt results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --endpoint1_ckpt results/vgg16/cifar10/endpoints/checkpoints/seed1/checkpoint-200.pt \
    --output results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \
    --dataset CIFAR10 \
    --data_path ./data \
    --model VGG16 \
    --transform VGG \
    --curve Bezier \
    --num_bends 3 \
    --num_points 61 \
    --batch_size 128 \
    --num_workers 4 \
    --use_test

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Detailed evaluation failed!"
    exit 1
fi

echo ""
echo "========================================"
echo "EVALUATION COMPLETED SUCCESSFULLY"
echo "========================================"
echo ""
echo "Results saved to:"
echo "  results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz"
echo ""
echo "To download results to local machine:"
echo "  scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz \\"
echo "    results/vgg16/cifar10/curve/evaluations/"
echo ""
echo "Next steps (run locally):"
echo "  1. Download predictions_detailed.npz"
echo "  2. Run analysis: python scripts/analysis/analyze_prediction_changes.py"
echo "  3. Create visualizations: python scripts/plot/plot_prediction_changes.py"
echo "  4. Create animation: python scripts/plot/create_prediction_animation.py"
echo ""
