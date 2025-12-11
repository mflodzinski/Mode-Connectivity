#!/bin/bash
# Calculate L2 distances for all endpoint pairs
# This shows the weight space distance between modes before curve training

set -e

echo "=========================================="
echo "Calculating L2 Distances for All Endpoints"
echo "=========================================="

# Create output directory
OUTPUT_DIR="results/vgg16/cifar10/endpoint_l2_distances"
mkdir -p "$OUTPUT_DIR"

# Endpoints
SEED0="results/vgg16/cifar10/endpoints/standard/seed0/checkpoint-200.pt"
SEED1="results/vgg16/cifar10/endpoints/standard/seed1/checkpoint-200.pt"
MIRROR="results/vgg16/cifar10/endpoints/standard/seed0_mirrored/checkpoint-200.pt"
EARLY2="results/vgg16/cifar10/endpoints/neuronswap/early2/checkpoint-200.pt"
MID2="results/vgg16/cifar10/endpoints/neuronswap/mid2/checkpoint-200.pt"
LATE2="results/vgg16/cifar10/endpoints/neuronswap/late2/checkpoint-200.pt"

echo ""
echo "Endpoints:"
echo "  - seed0: $SEED0"
echo "  - seed1: $SEED1"
echo "  - mirror: $MIRROR"
echo "  - early2: $EARLY2"
echo "  - mid2: $MID2"
echo "  - late2: $LATE2"
echo ""

# Function to calculate L2 distance
calculate_l2() {
    local name=$1
    local ckpt1=$2
    local ckpt2=$3
    local output="${OUTPUT_DIR}/${name}.json"

    if [ ! -f "$ckpt1" ]; then
        echo "⚠ Skipping $name: $ckpt1 not found"
        return
    fi

    if [ ! -f "$ckpt2" ]; then
        echo "⚠ Skipping $name: $ckpt2 not found"
        return
    fi

    echo "=========================================="
    echo "Calculating: $name"
    echo "=========================================="

    python scripts/analysis/calculate_endpoint_l2.py \
        --checkpoint1 "$ckpt1" \
        --checkpoint2 "$ckpt2" \
        --output "$output" \
        --show-top-k 5

    echo ""
}

# Calculate all pairwise distances
echo "1. SEED0 ↔ SEED1 (Different random initializations)"
calculate_l2 "seed0-seed1" "$SEED0" "$SEED1"

if [ -f "$MIRROR" ]; then
    echo "2. SEED0 ↔ MIRROR (Full neuron permutation)"
    calculate_l2 "seed0-mirror" "$SEED0" "$MIRROR"
fi

if [ -f "$EARLY2" ]; then
    echo "3. SEED0 ↔ EARLY2 (2 neurons swapped in early layer)"
    calculate_l2 "seed0-early2" "$SEED0" "$EARLY2"
fi

if [ -f "$MID2" ]; then
    echo "4. SEED0 ↔ MID2 (2 neurons swapped in mid layer)"
    calculate_l2 "seed0-mid2" "$SEED0" "$MID2"
fi

if [ -f "$LATE2" ]; then
    echo "5. SEED0 ↔ LATE2 (2 neurons swapped in late layer)"
    calculate_l2 "seed0-late2" "$SEED0" "$LATE2"
fi

# Create summary comparison
echo "=========================================="
echo "Creating Summary Comparison"
echo "=========================================="

SUMMARY_FILE="${OUTPUT_DIR}/summary.txt"

cat > "$SUMMARY_FILE" << 'EOF'
L2 Distance Summary Between Endpoints
======================================

This file summarizes the weight space distances between different endpoint pairs.

EOF

# Extract and compare normalized L2 distances
for json_file in "$OUTPUT_DIR"/*.json; do
    if [ -f "$json_file" ]; then
        name=$(basename "$json_file" .json)
        norm_l2=$(python -c "import json; print(json.load(open('$json_file'))['normalized_total_l2'])")
        total_l2=$(python -c "import json; print(json.load(open('$json_file'))['total_l2'])")
        echo "$name: normalized_l2=$norm_l2, total_l2=$total_l2" >> "$SUMMARY_FILE"
    fi
done

cat "$SUMMARY_FILE"

echo ""
echo "=========================================="
echo "Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files created:"
ls -lh "$OUTPUT_DIR"
echo ""
echo "Summary: $SUMMARY_FILE"
echo ""
echo "=========================================="
echo "EXPECTED PATTERNS:"
echo "=========================================="
echo ""
echo "seed0-seed1:    LARGE distance (different initialization)"
echo "seed0-mirror:   LARGE distance (all neurons permuted)"
echo "seed0-early2:   SMALL distance (only 2 neurons swapped)"
echo "seed0-mid2:     SMALL distance (only 2 neurons swapped)"
echo "seed0-late2:    SMALL distance (only 2 neurons swapped)"
echo ""
echo "The neuron swap distances should be orders of magnitude"
echo "smaller than the different initialization distance,"
echo "demonstrating minimal perturbation in weight space."
echo "=========================================="
