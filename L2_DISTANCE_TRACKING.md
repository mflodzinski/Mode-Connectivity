# L2 Distance Tracking Between Neural Network Modes

## Overview

This document describes the L2 distance calculation and tracking functionality added to the neuron swapping experiments. L2 distance measures the Euclidean distance between two neural networks in weight space.

## Why Track L2 Distance?

**L2 distance** answers: "How far apart are two networks in weight space?"

This is important because:
1. **Quantifies perturbation size**: Shows how much weight space changes with neuron swaps
2. **Baseline for connectivity**: Establishes the distance the curve must traverse
3. **Comparison across experiments**: Enables comparing:
   - 2-neuron swap vs full permutation
   - Early vs mid vs late layer swaps
   - Different initialization (seed0-seed1) vs permutation (seed0-mirror)

### Key Insight

**Functionally equivalent networks** (same accuracy) can have:
- **Small L2 distance**: Neuron swaps (2 neurons) → minimal weight space perturbation
- **Large L2 distance**: Full permutations (all neurons) → major weight space change

Yet both require finding connectivity paths!

---

## Where L2 Distance is Calculated

### 1. **During Neuron Swapping** (`swap_neurons.py`)

**When:** Immediately after creating swapped network
**What:** Compares original vs swapped weights
**Output:**
- Console display during execution
- `checkpoint-200_metadata.json` - Summary with top 5 layers
- `checkpoint-200_layer_distances.json` - Complete per-layer breakdown

**Example output:**
```
L2 Distance Statistics:
  Total L2 distance:      45.234567
  Normalized L2 distance: 0.003210
  Total parameters:       14,728,266

Top 5 layers by normalized L2 distance:
  1. layer_blocks.2.1.weight: 0.125634 (raw: 234.56, params: 589,824)
  2. classifier.1.weight: 0.089432 (raw: 186.43, params: 262,144)
  ...
```

### 2. **Before Curve Training** (`run_garipov_curve.py`)

**When:** Before starting curve training (startup)
**What:** Calculates distance between the two endpoints that will be connected
**Output:**
- Console display
- `endpoint_l2_distance.txt` in curve output directory

**Example:**
```
L2 Distance Between Endpoints
==================================================

Endpoint 0: results/vgg16/cifar10/endpoints/.../seed0/checkpoint-200.pt
Endpoint 1: results/vgg16/cifar10/endpoints_neuronswap/mid2/checkpoint-200.pt

Total L2 distance:      45.234567
Normalized L2 distance: 0.003210
Total parameters:       14,728,266
```

### 3. **Batch Calculation** (`calculate_all_endpoint_l2.sh`)

**When:** On-demand analysis of all endpoint pairs
**What:** Calculates L2 for all combinations
**Output:**
- Individual JSON files for each pair
- `summary.txt` with comparison table

---

## Usage

### Quick: Single Pair Calculation

```bash
python scripts/analysis/calculate_endpoint_l2.py \
    --checkpoint1 results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --checkpoint2 results/vgg16/cifar10/endpoints_neuronswap/mid2/checkpoint-200.pt \
    --output results/l2_distance.json \
    --show-top-k 10 \
    --sort-by normalized_l2
```

**Options:**
- `--checkpoint1`, `--checkpoint2`: Paths to checkpoints
- `--output`: Save results as JSON (optional)
- `--show-top-k`: Number of top layers to display (default: 10)
- `--sort-by`: Metric to sort by (`normalized_l2`, `raw_l2`, `relative_distance`)

### Automatic: All Pairs

```bash
bash scripts/workflows/calculate_all_endpoint_l2.sh
```

This calculates:
- seed0 ↔ seed1 (different initialization)
- seed0 ↔ mirror (full permutation)
- seed0 ↔ early2 (2 neurons, early layer)
- seed0 ↔ mid2 (2 neurons, mid layer)
- seed0 ↔ late2 (2 neurons, late layer)

**Output:** `results/vgg16/cifar10/endpoint_l2_distances/`

### Integrated: During Experiments

L2 distances are **automatically calculated** when you:
1. Create swapped endpoints (`setup_neuronswap_experiments.sh`)
2. Train curves (via `run_garipov_curve.py`)

No extra steps needed!

---

## Distance Metrics Explained

### 1. **Total L2 Distance** (Raw L2)

$$\text{Total L2} = \sqrt{\sum_{i=1}^{N} (w_i^{(1)} - w_i^{(2)})^2}$$

- **Units:** Absolute distance in weight space
- **Interpretation:** Total Euclidean distance across all parameters
- **Problem:** Biased toward large networks (more parameters = larger distance)

### 2. **Normalized L2 Distance** ⭐ (Recommended)

$$\text{Normalized L2} = \frac{\text{Total L2}}{\sqrt{N_{params}}}$$

- **Units:** Per-parameter RMS distance
- **Interpretation:** "Average" distance per parameter
- **Advantage:** Fair comparison across different model sizes
- **Use case:** Primary metric for comparing experiments

### 3. **Relative Distance** (Percentage)

$$\text{Relative} = \frac{\|W^{(1)} - W^{(2)}\|_2}{\|W^{(1)}\|_2}$$

- **Units:** Percentage of original weight magnitude
- **Interpretation:** Proportional change
- **Use case:** Understanding magnitude of perturbation

---

## Expected Results

### Hypothesis: Neuron Swaps Create Minimal Perturbation

| Experiment | Expected Normalized L2 | Interpretation |
|------------|------------------------|----------------|
| **seed0 ↔ seed1** | **~1.5 - 3.0** | Large (different initializations) |
| **seed0 ↔ mirror** | **~1.0 - 2.0** | Large (full permutation) |
| **seed0 ↔ early2** | **~0.001 - 0.01** | **Tiny** (2 neurons) |
| **seed0 ↔ mid2** | **~0.001 - 0.01** | **Tiny** (2 neurons) |
| **seed0 ↔ late2** | **~0.001 - 0.01** | **Tiny** (2 neurons) |

**Key finding:** 2-neuron swaps should be **orders of magnitude smaller** than full permutation or different initialization.

### Layer-wise Analysis

For 2-neuron swaps, expect:
- **Swapped layer:** High distance
- **Adjacent layer(s):** Moderate distance (due to chaining)
- **Distant layers:** Near-zero distance

This pattern reveals whether the perturbation is:
- **Local**: Only swapped + adjacent layers show distance
- **Global**: Many layers show significant distance

---

## Files Generated

### Per-Experiment Files

**Neuron swap endpoint:**
```
results/vgg16/cifar10/endpoints_neuronswap/mid2/
├── checkpoint-200.pt                    # Swapped model
├── checkpoint-200_metadata.json         # Includes L2 summary
└── checkpoint-200_layer_distances.json  # Complete per-layer L2
```

**Curve training:**
```
results/vgg16/cifar10/curve_neuronswap_mid2_reg/checkpoints/
└── endpoint_l2_distance.txt             # L2 between endpoints
```

### Batch Analysis Files

```
results/vgg16/cifar10/endpoint_l2_distances/
├── seed0-seed1.json           # L2 distance + per-layer breakdown
├── seed0-mirror.json
├── seed0-early2.json
├── seed0-mid2.json
├── seed0-late2.json
└── summary.txt                # Quick comparison table
```

---

## Interpreting Results

### Small Distance (< 0.1)
**Interpretation:** Networks are close in weight space
- Likely: Permutation symmetry or small perturbation
- Curve training should be easier
- Path may be nearly linear

### Medium Distance (0.1 - 1.0)
**Interpretation:** Networks are distinguishable but related
- Possible: Different training trajectories from same initialization
- Curve training requires optimization

### Large Distance (> 1.0)
**Interpretation:** Networks are far apart in weight space
- Likely: Different random initializations
- Curve training is challenging
- Non-linear path necessary

### ⚠️ Important Caveat

**Weight space distance ≠ Functional distance**

Two networks can be:
- **Far in weight space** (large L2)
- **Identical functionally** (same predictions)

Example: Permuted networks have large weight distance but 100% identical outputs!

---

## Research Questions Answered

### Q1: How much does a 2-neuron swap perturb weight space?

**Answer:** Check normalized L2 distance in metadata JSON
**Expected:** ~0.001 - 0.01 (orders of magnitude smaller than different initialization)

### Q2: Does layer depth affect perturbation size?

**Answer:** Compare:
- `seed0-early2.json` vs
- `seed0-mid2.json` vs
- `seed0-late2.json`

**Hypothesis:** Similar distances (same number of swaps), but layer-wise breakdown may differ

### Q3: How does 2-neuron swap compare to full permutation?

**Answer:** Compare:
- `seed0-mid2.json` (2 neurons) vs
- `seed0-mirror.json` (all neurons)

**Expected:** Mirror distance is **100-1000x larger**

### Q4: Which layers contribute most to the distance?

**Answer:** Check `top_5_layers` in metadata JSON
**Expected:** For 2-neuron swap, swapped layer dominates

---

## Integration with Mode Connectivity Analysis

The L2 distance provides context for interpreting connectivity results:

1. **Baseline distance:** Shows what the curve must traverse
2. **Barrier height vs distance:** Compare:
   - L2 distance (weight space)
   - Loss barrier height (function space)

3. **Efficiency:** Curves that traverse long distances with low barriers indicate smooth loss landscape

**Example interpretation:**
```
Experiment: seed0 ↔ mid2
L2 distance: 0.0032
Loss barrier: 0.05 (5% error increase)
Conclusion: Very small perturbation, minimal barrier → smooth landscape
```

---

## Troubleshooting

### Issue: Distance is zero or very small for seed0-seed1

**Problem:** You're comparing the same checkpoint twice
**Solution:** Verify checkpoint paths are different

### Issue: Distance is huge for neuron swap

**Problem:** Swap script may have failed; models aren't functionally equivalent
**Solution:** Check verification results in metadata

### Issue: Layer distances don't sum correctly

**Expected:** This is normal; we use L2 norm, not L1
**Explanation:** Total L2 = sqrt(sum of squared layer L2s), not sum of layer L2s

---

## Command Reference

### Create swapped endpoint with L2 calculation
```bash
python scripts/analysis/swap_neurons.py \
    --checkpoint ORIGINAL.pt \
    --output SWAPPED.pt \
    --layer-depth mid \
    --num-swaps 1 \
    --verify
# L2 distance automatically calculated and saved
```

### Calculate L2 between any two checkpoints
```bash
python scripts/analysis/calculate_endpoint_l2.py \
    --checkpoint1 FIRST.pt \
    --checkpoint2 SECOND.pt \
    --output results.json
```

### Calculate all endpoint pairs
```bash
bash scripts/workflows/calculate_all_endpoint_l2.sh
```

### Check L2 distance for curve
```bash
# After curve training, check:
cat results/vgg16/cifar10/curve_neuronswap_mid2_reg/checkpoints/endpoint_l2_distance.txt
```

---

## Advanced: Adding L2 Tracking to New Experiments

If you create new swapping methods, add L2 calculation:

```python
from scripts.analysis.swap_neurons import calculate_l2_distance

# After creating modified model
l2_stats = calculate_l2_distance(original_model, modified_model)

# Log results
print(f"L2 distance: {l2_stats['total_l2']:.6f}")
print(f"Normalized: {l2_stats['normalized_total_l2']:.6f}")

# Save to metadata
metadata['l2_distance'] = {
    'total_l2': float(l2_stats['total_l2']),
    'normalized_total_l2': float(l2_stats['normalized_total_l2'])
}
```

---

## Summary

✅ **What was added:**
- L2 calculation in `swap_neurons.py`
- Standalone utility `calculate_endpoint_l2.py`
- Batch analysis script `calculate_all_endpoint_l2.sh`
- Automatic L2 logging in `run_garipov_curve.py`

✅ **When it runs:**
- Creating swapped endpoints
- Training curves
- On-demand analysis

✅ **Output:**
- Console display (immediate feedback)
- Metadata JSON (per-experiment)
- Standalone JSON files (batch analysis)
- Text summaries (human-readable)

✅ **Key metric:**
- **Normalized L2**: Fair comparison across experiments

**Use L2 distance to validate that neuron swaps create minimal perturbations while maintaining functional equivalence!**
