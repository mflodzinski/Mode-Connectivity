#!/bin/bash
# Run prediction changes analysis (Steps 2-4) - LOCAL ONLY
# Requires: predictions_detailed.npz downloaded from cluster

set -e  # Exit on error

PREDICTIONS_FILE="results/vgg16/cifar10/curve/evaluations/predictions_detailed.npz"
ANALYSIS_DIR="results/vgg16/cifar10/curve/analysis"
FIGURES_DIR="results/vgg16/cifar10/curve/figures"

echo ""
echo "========================================"
echo "PREDICTION CHANGES ANALYSIS PIPELINE"
echo "========================================"
echo ""

# Check if predictions file exists
if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "ERROR: Predictions file not found!"
    echo "Expected location: $PREDICTIONS_FILE"
    echo ""
    echo "Please download it from the cluster first:"
    echo "  scp mlodzinski@login.daic.tudelft.nl:~/Mode-Connectivity/$PREDICTIONS_FILE \\"
    echo "    $PREDICTIONS_FILE"
    echo ""
    exit 1
fi

echo "✓ Found predictions file: $PREDICTIONS_FILE"
echo ""

# Step 2: Analyze prediction changes
echo "========================================"
echo "STEP 2: Analyzing Prediction Changes"
echo "========================================"
echo ""

poetry run python scripts/analysis/analyze_prediction_changes.py \
    --predictions "$PREDICTIONS_FILE" \
    --output "$ANALYSIS_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Analysis failed!"
    exit 1
fi

echo ""
echo "✓ Analysis complete"
echo ""

# Step 3: Create 2D visualization
echo "========================================"
echo "STEP 3: Creating 2D Visualization"
echo "========================================"
echo ""

poetry run python scripts/plot/plot_prediction_changes.py \
    --analysis "$ANALYSIS_DIR/changing_samples_info.npz" \
    --output "$FIGURES_DIR" \
    --method umap \
    --random_state 42

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Visualization failed!"
    exit 1
fi

echo ""
echo "✓ 2D visualization complete"
echo ""

# Step 4: Create animation
echo "========================================"
echo "STEP 4: Creating Animation (GIF)"
echo "========================================"
echo ""
echo "This may take a few minutes..."
echo ""

poetry run python scripts/plot/create_prediction_animation.py \
    --analysis "$ANALYSIS_DIR/changing_samples_info.npz" \
    --predictions "$PREDICTIONS_FILE" \
    --output "$FIGURES_DIR/prediction_evolution.gif" \
    --method umap \
    --fps 5 \
    --random_state 42 \
    --skip_frames 1 <<< "n"  # Auto-answer 'n' to frame deletion prompt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Animation creation failed!"
    exit 1
fi

echo ""
echo "✓ Animation complete"
echo ""

# Summary
echo "========================================"
echo "ALL ANALYSIS STEPS COMPLETED!"
echo "========================================"
echo ""
echo "Generated files:"
echo "  Analysis:"
echo "    - $ANALYSIS_DIR/changing_samples_info.npz"
echo "    - $ANALYSIS_DIR/analysis_summary.txt"
echo ""
echo "  Visualizations:"
echo "    - $FIGURES_DIR/prediction_changes_2d.png"
echo "    - $FIGURES_DIR/prediction_evolution.gif"
echo ""
echo "View results:"
echo "  open $FIGURES_DIR/prediction_changes_2d.png"
echo "  open $FIGURES_DIR/prediction_evolution.gif"
echo "  cat $ANALYSIS_DIR/analysis_summary.txt"
echo ""
