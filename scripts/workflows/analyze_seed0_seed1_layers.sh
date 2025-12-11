#!/bin/bash
# Analyze layer-wise distances for seed0-seed1 curve

set -e

echo "=========================================="
echo "Analyzing seed0-seed1 Layer Distances"
echo "=========================================="

# Paths
CURVE_CHECKPOINT="results/vgg16/cifar10/curves/standard/seed0-seed1_noreg/checkpoints/checkpoint-200.pt"
ENDPOINT0="results/vgg16/cifar10/endpoints/standard/seed0/checkpoint-200.pt"
ENDPOINT1="results/vgg16/cifar10/endpoints/standard/seed1/checkpoint-200.pt"
OUTPUT_DIR="results/vgg16/cifar10/curves/standard/seed0-seed1_noreg/layer_analysis"

# Create temporary metadata file (no swaps for seed0-seed1)
TMP_METADATA=$(mktemp)
cat > "$TMP_METADATA" << 'EOF'
{
  "experiment_type": "different_initialization",
  "layer_depth": "none",
  "layer_description": "No neuron swaps - different random initializations",
  "block_idx": -1,
  "layer_idx": -1,
  "num_swaps": 0,
  "neuron_pairs": []
}
EOF

echo ""
echo "Running analysis..."
python scripts/analysis/analyze_neuron_swap_distances.py \
    --curve-checkpoint "$CURVE_CHECKPOINT" \
    --original-checkpoint "$ENDPOINT0" \
    --swap-metadata "$TMP_METADATA" \
    --output-dir "$OUTPUT_DIR" \
    --num-points 61

# Cleanup
rm "$TMP_METADATA"

echo ""
echo "=========================================="
echo "Creating visualization..."
echo "=========================================="

python scripts/plotting/plot_layer_distance_animation.py \
    --data "$OUTPUT_DIR/layer_distances_along_curve.npz" \
    --output "$OUTPUT_DIR/layer_distances_evolution.gif" \
    --heatmap

echo ""
echo "=========================================="
echo "COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "View animation:"
echo "  open $OUTPUT_DIR/layer_distances_evolution.gif"
echo ""
echo "=========================================="
