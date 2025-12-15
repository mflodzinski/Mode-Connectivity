#!/bin/bash
# Setup neuron swap experiments
# This script creates swapped endpoints for different layer depths

set -e  # Exit on error

echo "=========================================="
echo "Setting up Neuron Swap Experiments"
echo "=========================================="

# Directories
ENDPOINT_DIR="results/vgg16/cifar10/endpoints/standard/seed0"
SWAP_DIR="results/vgg16/cifar10/endpoints/neuronswap"
CHECKPOINT="${ENDPOINT_DIR}/checkpoint-200.pt"

# Check if original checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Original checkpoint not found at $CHECKPOINT"
    exit 1
fi

echo ""
echo "Original checkpoint: $CHECKPOINT"
echo "Output directory: $SWAP_DIR"
echo ""

# Create output directories
mkdir -p "${SWAP_DIR}/early2"
mkdir -p "${SWAP_DIR}/mid2"
mkdir -p "${SWAP_DIR}/late2"

# 1. Early layer (2 neurons)
echo "=========================================="
echo "1. Creating early layer swap (2 neurons)"
echo "=========================================="
python scripts/analysis/swap_neurons.py \
    --checkpoint "$CHECKPOINT" \
    --output "${SWAP_DIR}/early2/checkpoint-200.pt" \
    --layer-depth early \
    --num-swaps 1 \
    --seed 42 \
    --verify

# 2. Mid layer (2 neurons)
echo ""
echo "=========================================="
echo "2. Creating mid layer swap (2 neurons)"
echo "=========================================="
python scripts/analysis/swap_neurons.py \
    --checkpoint "$CHECKPOINT" \
    --output "${SWAP_DIR}/mid2/checkpoint-200.pt" \
    --layer-depth mid \
    --num-swaps 1 \
    --seed 42 \
    --verify

# 3. Late layer (2 neurons)
echo ""
echo "=========================================="
echo "3. Creating late layer swap (2 neurons)"
echo "=========================================="
python scripts/analysis/swap_neurons.py \
    --checkpoint "$CHECKPOINT" \
    --output "${SWAP_DIR}/late2/checkpoint-200.pt" \
    --layer-depth late \
    --num-swaps 1 \
    --seed 42 \
    --verify

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Created swapped endpoints:"
echo "  - ${SWAP_DIR}/early2/checkpoint-200.pt"
echo "  - ${SWAP_DIR}/mid2/checkpoint-200.pt"
echo "  - ${SWAP_DIR}/late2/checkpoint-200.pt"
echo ""
echo "Metadata files:"
echo "  - ${SWAP_DIR}/early2/checkpoint-200_metadata.json"
echo "  - ${SWAP_DIR}/mid2/checkpoint-200_metadata.json"
echo "  - ${SWAP_DIR}/late2/checkpoint-200_metadata.json"
echo ""
echo "Next steps:"
echo "  1. Review metadata files to see which neurons were swapped"
echo "  2. Train curves using SLURM scripts:"
echo "     sbatch scripts/slurm/submit_neuronswap_curve_early2.sh"
echo "     sbatch scripts/slurm/submit_neuronswap_curve_mid2.sh"
echo "     sbatch scripts/slurm/submit_neuronswap_curve_late2.sh"
echo "  3. After training, analyze with:"
echo "     python scripts/analysis/analyze_neuron_swap_distances.py"
echo "=========================================="
