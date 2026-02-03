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
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Checkpoints
W0="results/vgg16/cifar10/endpoints/lmc_connected/seed0/checkpoint-200.pt"
W1="results/vgg16/cifar10/endpoints/lmc_connected/seed1/checkpoint-200.pt"
OUTPUT_DIR="results/vgg16/cifar10/endpoints/lmc_connected/evaluations"

mkdir -p $OUTPUT_DIR

# Evaluate linear interpolation using eval_linear.py
echo "Evaluating linear interpolation between w_0 and w_1..."
srun python scripts/eval/eval_linear.py \
    --dir $OUTPUT_DIR \
    --init_start $W0 \
    --init_end $W1 \
    --num_points 21 \
    --dataset CIFAR10 \
    --data_path ./data \
    --model VGG16 \
    --transform VGG \
    --batch_size 128 \
    --use_test

echo ""
echo "========================================"
echo "EVALUATION COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_DIR/linear.npz"
echo ""
echo "To download:"
echo "  scp mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/$OUTPUT_DIR/linear.npz ."
