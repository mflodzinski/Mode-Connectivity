#!/bin/bash
# Save images of the top 20 most unstable samples

set -e

echo "=========================================="
echo "Saving Edge Case Sample Images"
echo "=========================================="

# Paths
PREDICTIONS_FILE="results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/predictions_detailed.npz"
OUTPUT_DIR="results/vgg16/cifar10/curves/standard/seed0-seed1_reg/edge_cases/images"

if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "Error: Predictions file not found: $PREDICTIONS_FILE"
    exit 1
fi

echo ""
echo "Input: $PREDICTIONS_FILE"
echo "Output: $OUTPUT_DIR"
echo ""

poetry run python scripts/analysis/save_unstable_sample_images.py \
    --predictions-file "$PREDICTIONS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --top-k 20

echo ""
echo "=========================================="
echo "View images:"
echo "  ls -lh $OUTPUT_DIR"
echo "  open $OUTPUT_DIR"
echo "=========================================="
