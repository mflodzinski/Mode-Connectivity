#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_permutation_path_alignment_%j.out
#SBATCH --error=slurm_permutation_path_alignment_%j.err
#SBATCH --job-name=perm_path_align
#SBATCH --gres=gpu:a40:1

# Permutation-only endpoint alignment on seed0/seed1 via low-loss path samples
# Trains a PolyChain path, samples C0..C4, runs all baselines, and saves comparison outputs.

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Create output directory
mkdir -p results/vgg16/cifar10/alignment/permutation_path

ENDPOINT_A="results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt"
ENDPOINT_B="results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt"
OUTPUT_ROOT="results/vgg16/cifar10/alignment/permutation_path"
EXPERIMENT_NAME="seed0-seed1_polychain_path_alignment"

echo "========================================"
echo "Permutation Path Alignment"
echo "========================================"
echo "endpoint_a: $ENDPOINT_A"
echo "endpoint_b: $ENDPOINT_B"
echo "output_root: $OUTPUT_ROOT"
echo "experiment_name: $EXPERIMENT_NAME"
echo ""

CMD=(
    python scripts/analysis/run_permutation_path_alignment.py
    endpoint_a="$ENDPOINT_A"
    endpoint_b="$ENDPOINT_B"
    output_root="$OUTPUT_ROOT"
    experiment_name="$EXPERIMENT_NAME"
    overwrite=true
)

srun "${CMD[@]}"

echo ""
echo "========================================"
echo "PIPELINE COMPLETE!"
echo "========================================"
echo ""
echo "Results saved to: $OUTPUT_ROOT/$EXPERIMENT_NAME"
