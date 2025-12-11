#!/bin/bash
# Re-run layer-wise distance analysis on neuron swap curves with new metric
# (difference of L2 norms instead of L2 norm of differences)

echo "================================================================================"
echo "RE-ANALYZING NEURON SWAP CURVES WITH NEW METRIC"
echo "================================================================================"
echo ""
echo "New metric: ||w_t||₂ - ||w_0||₂ (difference of L2 norms)"
echo "Previous metric: ||w_t - w_0||₂ (L2 norm of differences)"
echo ""
echo "This change makes the metric permutation-invariant:"
echo "- Swapped layers will now show ≈0 at both t=0 and t=1"
echo "- Better demonstrates that neuron swapping creates functional equivalence"
echo ""
echo "================================================================================"

# Original endpoint (after reorganization)
ORIGINAL_CHECKPOINT="results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt"

# Early2 neuron swap
echo ""
echo "--------------------------------------------------------------------------------"
echo "1. EARLY2 NEURON SWAP (Block 0, Conv 1)"
echo "--------------------------------------------------------------------------------"

poetry run python scripts/analysis/analyze_neuron_swap_distances.py \
  --curve-checkpoint results/vgg16/cifar10/curves/neuronswap/early2_reg/checkpoints/checkpoint-30.pt \
  --original-checkpoint ${ORIGINAL_CHECKPOINT} \
  --swap-metadata results/vgg16/cifar10/endpoints/neuronswap/early2/analysis/checkpoint-200_metadata.json \
  --output-dir results/vgg16/cifar10/curves/neuronswap/early2_reg/analysis \
  --num-points 61

if [ $? -eq 0 ]; then
    echo "✓ Early2 analysis complete"
else
    echo "✗ Early2 analysis failed"
fi

# Mid2 neuron swap
echo ""
echo "--------------------------------------------------------------------------------"
echo "2. MID2 NEURON SWAP (Block 2, Conv 1)"
echo "--------------------------------------------------------------------------------"

poetry run python scripts/analysis/analyze_neuron_swap_distances.py \
  --curve-checkpoint results/vgg16/cifar10/curves/neuronswap/mid2_reg/checkpoints/checkpoint-30.pt \
  --original-checkpoint ${ORIGINAL_CHECKPOINT} \
  --swap-metadata results/vgg16/cifar10/endpoints/neuronswap/mid2/analysis/checkpoint-200_metadata.json \
  --output-dir results/vgg16/cifar10/curves/neuronswap/mid2_reg/analysis \
  --num-points 61

if [ $? -eq 0 ]; then
    echo "✓ Mid2 analysis complete"
else
    echo "✗ Mid2 analysis failed"
fi

# Late2 neuron swap
echo ""
echo "--------------------------------------------------------------------------------"
echo "3. LATE2 NEURON SWAP (Block 4, Conv 1)"
echo "--------------------------------------------------------------------------------"

poetry run python scripts/analysis/analyze_neuron_swap_distances.py \
  --curve-checkpoint results/vgg16/cifar10/curves/neuronswap/late2_reg/checkpoints/checkpoint-30.pt \
  --original-checkpoint ${ORIGINAL_CHECKPOINT} \
  --swap-metadata results/vgg16/cifar10/endpoints/neuronswap/late2/analysis/checkpoint-200_metadata.json \
  --output-dir results/vgg16/cifar10/curves/neuronswap/late2_reg/analysis \
  --num-points 61

if [ $? -eq 0 ]; then
    echo "✓ Late2 analysis complete"
else
    echo "✗ Late2 analysis failed"
fi

echo ""
echo "================================================================================"
echo "RE-ANALYSIS COMPLETE"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  - results/vgg16/cifar10/curves/neuronswap/early2_reg/analysis/"
echo "  - results/vgg16/cifar10/curves/neuronswap/mid2_reg/analysis/"
echo "  - results/vgg16/cifar10/curves/neuronswap/late2_reg/analysis/"
echo ""
echo "Next steps:"
echo "  1. Regenerate visualizations with:"
echo "     bash scripts/plot/regenerate_neuronswap_plots.sh"
echo ""
echo "  2. Compare old vs new results to verify swapped layers now show symmetry"
echo ""
echo "================================================================================"
