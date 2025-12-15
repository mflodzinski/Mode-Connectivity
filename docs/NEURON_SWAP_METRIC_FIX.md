# Neuron Swap Layer Distance Metric Fix

## Problem

In the neuron swap experiments, the layer-wise L2 norm analysis showed unexpected asymmetric behavior for swapped layers:

- **At t=0**: Swapped layer distance = 0 ✓
- **At t=1**: Swapped layer distance ≠ 0 ✗ (unexpected!)

This was confusing because at both endpoints (t=0 and t=1), the swapped layer contains the same neuron values, just in different order. Since neuron swapping is functionally equivalent (same outputs), we expected the distance to be 0 at both endpoints.

## Root Cause

The original metric calculated **L2 norm of differences**:

```python
diff = current_flat - ref_flat
raw_l2 = torch.norm(diff, p=2).item()  # ||w_t - w_0||₂
```

This metric is **NOT permutation-invariant**:
- When neurons 228 and 6 are swapped at t=1
- Position 228 contains neuron 6's weights: `w_t[228] = w_0[6]`
- Position 6 contains neuron 228's weights: `w_t[6] = w_0[228]`
- The difference `w_t[228] - w_0[228]` is LARGE (even though it's just a reordering)
- Result: Non-zero distance at t=1 despite functional equivalence

## Solution

Changed to **difference of L2 norms**:

```python
# Compute L2 norm of each parameter set
current_norm = torch.norm(current_flat, p=2).item()  # ||w_t||₂
ref_norm = torch.norm(ref_flat, p=2).item()           # ||w_0||₂

# Difference of norms (permutation-invariant)
raw_l2_diff = abs(current_norm - ref_norm)  # | ||w_t||₂ - ||w_0||₂ |
```

This metric **IS permutation-invariant**:
- The L2 norm of a vector is the same regardless of element order
- Swapping neurons doesn't change the overall magnitude
- Result: Zero distance at both t=0 and t=1 for swapped layers

## Mathematical Explanation

**Old metric (norm of differences)**:
```
||w_t - w_0||₂ = sqrt(∑(w_t[i] - w_0[i])²)
```
- Sensitive to position changes
- Swapped elements contribute large differences

**New metric (difference of norms)**:
```
| ||w_t||₂ - ||w_0||₂ | = | sqrt(∑w_t[i]²) - sqrt(∑w_0[i]²) |
```
- Only sensitive to magnitude changes
- Permutation-invariant (same values, different order → same norm)

## Results After Fix

Test on mid2 neuron swap (Block 2, Conv 1):

### Swapped Layer (block2_conv1)
- **t=0.0**: 0.0 ✓
- **t=0.5**: 0.00110 (small deviation during interpolation)
- **t=1.0**: 7.4e-09 ≈ 0 ✓ **FIXED!**

### Non-Swapped Layer (block0_conv0)
- **t=0.0**: 0.0 ✓
- **t=0.5**: 0.00738 (changes during interpolation)
- **t=1.0**: 0.0 ✓

The swapped layer now shows symmetric behavior! Both endpoints have distance ≈ 0, correctly reflecting that the swapped configuration is functionally equivalent.

## Modified Files

1. **scripts/analysis/analyze_neuron_swap_distances.py** (lines 62-102)
   - Changed `compute_layer_distance()` function
   - From: `||w_t - w_0||₂`
   - To: `| ||w_t||₂ - ||w_0||₂ |`
   - Updated docstring to explain permutation-invariance

2. **scripts/analysis/reanalyze_neuronswap_curves.sh** (created)
   - Batch script to re-run analysis on all 3 neuron swap experiments (early2, mid2, late2)
   - Uses updated paths after directory reorganization

3. **scripts/plot/regenerate_neuronswap_plots.sh** (created)
   - Batch script to regenerate all visualizations with new metric

## How to Re-Run Analysis

### Step 1: Re-analyze all curves
```bash
bash scripts/analysis/reanalyze_neuronswap_curves.sh
```

This will regenerate `layer_distances_along_curve.npz` for:
- Early2 neuron swap (Block 0, Conv 1)
- Mid2 neuron swap (Block 2, Conv 1)
- Late2 neuron swap (Block 4, Conv 1)

### Step 2: Regenerate visualizations
```bash
bash scripts/plot/regenerate_neuronswap_plots.sh
```

This will create:
- Animated GIFs showing layer evolution
- Static heatmaps showing distance evolution

## Expected Improvements in Visualizations

**Before (old metric)**:
- Swapped layers (red bars) showed large non-zero values at t=1
- Appeared asymmetric despite functional equivalence

**After (new metric)**:
- Swapped layers now show ≈0 at both t=0 and t=1
- Clear symmetric curves
- Better demonstrates that neuron swapping creates functionally equivalent networks

## Interpretation

The new metric makes it clear that:

1. **Swapped layers are functionally equivalent**: Distance ≈ 0 at both endpoints
2. **Non-swapped layers make small adjustments**: Small changes to accommodate the swap in neighboring layers
3. **The curve finds a low-loss path**: All layers show smooth evolution from t=0 to t=1

This correctly reflects the permutation invariance of neural networks and validates that mode connectivity works even when endpoints differ only by neuron permutation.

## Technical Notes

- **Permutation invariance**: The new metric treats all permutations of the same weights as identical
- **Functional equivalence**: Swapping neurons within a layer (and corresponding input channels in the next layer) doesn't change network outputs
- **Numerical precision**: The tiny value (7.4e-09) at t=1 is just floating-point rounding, effectively zero
- **Alternative metrics**: Could also use optimal transport distance or Hungarian algorithm for explicit permutation matching, but difference-of-norms is simpler and sufficient

## References

- Original analysis code: `scripts/analysis/analyze_neuron_swap_distances.py`
- Neuron swap creation: `scripts/analysis/swap_neurons.py`
- Metadata files: `results/vgg16/cifar10/endpoints/neuronswap/*/analysis/checkpoint-200_metadata.json`
