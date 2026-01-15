#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_eval_fmnist_polygon_%j.out
#SBATCH --error=slurm_eval_fmnist_polygon_%j.err
#SBATCH --job-name=eval_fmnist_polygon
#SBATCH --gres=gpu:a40:1

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

CHECKPOINT="results/convfc/fmnist/advanced_geometry/polygon/seed0-seed1/checkpoint-150.pt"
EVAL_DIR="results/convfc/fmnist/advanced_geometry/polygon/seed0-seed1/evaluations"

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi

echo "Evaluating Polygon: ${CHECKPOINT}"

# Create output directory
mkdir -p "${EVAL_DIR}"

# Run evaluation
srun python external/dnn-mode-connectivity/eval_curve.py \
  --dir "${EVAL_DIR}" \
  --dataset FashionMNIST \
  --data_path ./data \
  --transform VGG \
  --model ConvFC \
  --curve PolyChain \
  --num_bends 3 \
  --ckpt "${CHECKPOINT}" \
  --num_points 61 \
  --use_test

echo "Evaluation complete. Output saved to: ${EVAL_DIR}/curve.npz"
