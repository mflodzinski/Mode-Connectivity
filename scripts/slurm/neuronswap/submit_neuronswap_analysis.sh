#!/bin/bash
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --mail-type=END,FAIL
#SBATCH --output=slurm_neuronswap_analysis_%j.out
#SBATCH --error=slurm_neuronswap_analysis_%j.err
#SBATCH --job-name=neuronswap_analysis

# Activate virtual environment
source $HOME/venvs/mode-connectivity/bin/activate || . $HOME/venvs/mode-connectivity/bin/activate

# Navigate to project directory
cd /tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity

# Add project root to Python path
export PYTHONPATH=/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity:$PYTHONPATH

# Configuration - change these for different experiments
EXPERIMENT="mid2"  # Options: early2, mid2, late2

CURVE_CHECKPOINT="results/vgg16/cifar10/curve_neuronswap_${EXPERIMENT}_reg/checkpoints/checkpoint-200.pt"
ORIGINAL_CHECKPOINT="results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt"
SWAP_METADATA="results/vgg16/cifar10/endpoints_neuronswap/${EXPERIMENT}/checkpoint-200_metadata.json"
OUTPUT_DIR="results/vgg16/cifar10/curve_neuronswap_${EXPERIMENT}_reg/analysis"

echo "=========================================="
echo "Analyzing Neuron Swap Experiment: $EXPERIMENT"
echo "=========================================="
echo "Curve checkpoint: $CURVE_CHECKPOINT"
echo "Original checkpoint: $ORIGINAL_CHECKPOINT"
echo "Swap metadata: $SWAP_METADATA"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run analysis
srun python scripts/analysis/analyze_neuron_swap_distances.py \
    --curve-checkpoint "$CURVE_CHECKPOINT" \
    --original-checkpoint "$ORIGINAL_CHECKPOINT" \
    --swap-metadata "$SWAP_METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --num-points 61

echo ""
echo "=========================================="
echo "Creating Visualization"
echo "=========================================="

# Create animated GIF
srun python scripts/plotting/plot_layer_distance_animation.py \
    --data "${OUTPUT_DIR}/layer_distances_along_curve.npz" \
    --output "${OUTPUT_DIR}/layer_distances_evolution.gif" \
    --fps 10 \
    --metric normalized_l2 \
    --heatmap

echo ""
echo "=========================================="
echo "Analysis Complete!"
echo "=========================================="
echo "Results saved to: $OUTPUT_DIR"
echo "  - layer_distances_along_curve.npz"
echo "  - analysis_summary.json"
echo "  - layer_distances_evolution.gif"
echo "  - layer_distances_evolution_heatmap.png"
echo "=========================================="
