#!/bin/bash
# Regenerate all neuron swap visualizations with new metric

echo "================================================================================"
echo "REGENERATING NEURON SWAP VISUALIZATIONS"
echo "================================================================================"
echo ""

# Early2 neuron swap
echo "--------------------------------------------------------------------------------"
echo "1. EARLY2 NEURON SWAP"
echo "--------------------------------------------------------------------------------"

poetry run python scripts/plot/plot_layer_distance_animation.py \
  --data results/vgg16/cifar10/curves/neuronswap/early2_reg/analysis/layer_distances_along_curve.npz \
  --output results/vgg16/cifar10/curves/neuronswap/early2_reg/analysis/layer_distances_evolution.gif \
  --heatmap

if [ $? -eq 0 ]; then
    echo "✓ Early2 visualizations generated"
else
    echo "✗ Early2 visualizations failed"
fi

# Mid2 neuron swap
echo ""
echo "--------------------------------------------------------------------------------"
echo "2. MID2 NEURON SWAP"
echo "--------------------------------------------------------------------------------"

poetry run python scripts/plot/plot_layer_distance_animation.py \
  --data results/vgg16/cifar10/curves/neuronswap/mid2_reg/analysis/layer_distances_along_curve.npz \
  --output results/vgg16/cifar10/curves/neuronswap/mid2_reg/analysis/layer_distances_evolution.gif \
  --heatmap

if [ $? -eq 0 ]; then
    echo "✓ Mid2 visualizations generated"
else
    echo "✗ Mid2 visualizations failed"
fi

# Late2 neuron swap
echo ""
echo "--------------------------------------------------------------------------------"
echo "3. LATE2 NEURON SWAP"
echo "--------------------------------------------------------------------------------"

poetry run python scripts/plot/plot_layer_distance_animation.py \
  --data results/vgg16/cifar10/curves/neuronswap/late2_reg/analysis/layer_distances_along_curve.npz \
  --output results/vgg16/cifar10/curves/neuronswap/late2_reg/analysis/layer_distances_evolution.gif \
  --heatmap

if [ $? -eq 0 ]; then
    echo "✓ Late2 visualizations generated"
else
    echo "✗ Late2 visualizations failed"
fi

echo ""
echo "================================================================================"
echo "VISUALIZATION REGENERATION COMPLETE"
echo "================================================================================"
echo ""
echo "Generated files:"
echo "  Early2:"
echo "    - results/vgg16/cifar10/curves/neuronswap/early2_reg/analysis/layer_distances_evolution.gif"
echo "    - results/vgg16/cifar10/curves/neuronswap/early2_reg/analysis/layer_distances_evolution_heatmap.png"
echo ""
echo "  Mid2:"
echo "    - results/vgg16/cifar10/curves/neuronswap/mid2_reg/analysis/layer_distances_evolution.gif"
echo "    - results/vgg16/cifar10/curves/neuronswap/mid2_reg/analysis/layer_distances_evolution_heatmap.png"
echo ""
echo "  Late2:"
echo "    - results/vgg16/cifar10/curves/neuronswap/late2_reg/analysis/layer_distances_evolution.gif"
echo "    - results/vgg16/cifar10/curves/neuronswap/late2_reg/analysis/layer_distances_evolution_heatmap.png"
echo ""
echo "Expected changes:"
echo "  - Swapped layers (red bars) should now show ≈0 at both t=0 and t=1"
echo "  - Non-swapped layers should still show smooth symmetric curves"
echo ""
echo "================================================================================"
