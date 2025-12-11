#!/bin/bash
# Analyze layer-wise distances for seed0-mirror curve

set -e

echo "=========================================="
echo "Analyzing seed0-mirror Layer Distances"
echo "=========================================="

# Paths
CURVE_CHECKPOINT="results/vgg16/cifar10/curves/standard/seed0-mirror_noreg/checkpoints/checkpoint-200.pt"
ENDPOINT0="results/vgg16/cifar10/endpoints/standard/seed0/checkpoint-200.pt"
ENDPOINT1="results/vgg16/cifar10/endpoints/standard/seed0_mirrored/checkpoint-200.pt"
OUTPUT_DIR="results/vgg16/cifar10/curves/standard/seed0-mirror_noreg/layer_analysis"

# Create temporary metadata file (full permutation for mirror)
TMP_METADATA=$(mktemp)
cat > "$TMP_METADATA" << 'EOF'
{
  "experiment_type": "full_permutation",
  "layer_depth": "all",
  "layer_description": "Full neuron permutation (mirror)",
  "block_idx": -1,
  "layer_idx": -1,
  "num_swaps": 999999,
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
