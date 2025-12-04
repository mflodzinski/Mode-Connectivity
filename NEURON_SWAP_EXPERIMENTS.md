# Neuron Swap Experiments

## Overview

This experiment tests whether mode connectivity corrections are **local** or **global** when connecting networks that differ only by neuron permutations.

### Research Question

When we swap just 2 neurons in a network:
- **Hypothesis A (Local)**: Only the swapped neurons adjust along the connectivity path
- **Hypothesis B (Global)**: All weights readjust globally to accommodate the swap

This reveals fundamental properties of neural network weight space geometry.

---

## Experimental Design

### 1. Create Swapped Endpoints

Starting from a trained model (`seed0`), create three variants:
- **Early swap**: Swap 2 neurons in Block 0, Layer 0 (early features)
- **Mid swap**: Swap 2 neurons in Block 2, Layer 1 (mid-level features)
- **Late swap**: Swap 2 neurons in Block 4, Layer 1 (late features)

Each swapped network is **functionally equivalent** (same accuracy) but at a different point in weight space.

### 2. Train Bezier Curves

For each depth (early/mid/late), train a Bezier curve connecting:
- Start: Original network (seed0)
- End: Swapped network (seed0 with 2 neurons swapped)

### 3. Analyze Layer-wise Changes

Evaluate the curve at 61 points (t ∈ [0, 1]) and measure:
- **Per-layer distance** from original network
- **Normalized L2**: Distance scaled by number of parameters
- **Relative distance**: Percentage change in weights

### 4. Visualize Results

Create animated GIF showing how each layer changes along the curve.

---

## Quick Start

### Step 1: Create Swapped Endpoints

```bash
# Run the setup script
bash scripts/workflows/setup_neuronswap_experiments.sh
```

This creates:
- `results/vgg16/cifar10/endpoints_neuronswap/early2/checkpoint-200.pt`
- `results/vgg16/cifar10/endpoints_neuronswap/mid2/checkpoint-200.pt`
- `results/vgg16/cifar10/endpoints_neuronswap/late2/checkpoint-200.pt`

Each comes with metadata JSON showing which neurons were swapped.

### Step 2: Train Curves (on cluster)

```bash
# Submit curve training jobs
sbatch scripts/slurm/submit_neuronswap_curve_early2.sh
sbatch scripts/slurm/submit_neuronswap_curve_mid2.sh
sbatch scripts/slurm/submit_neuronswap_curve_late2.sh
```

Training time: ~2 hours per curve on A40 GPU

### Step 3: Analyze Results

```bash
# Edit the EXPERIMENT variable in the script (early2, mid2, or late2)
# Then submit analysis job
sbatch scripts/slurm/submit_neuronswap_analysis.sh
```

Or run locally:

```bash
# For mid-layer experiment
python scripts/analysis/analyze_neuron_swap_distances.py \
    --curve-checkpoint results/vgg16/cifar10/curve_neuronswap_mid2_reg/checkpoints/checkpoint-200.pt \
    --original-checkpoint results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt \
    --swap-metadata results/vgg16/cifar10/endpoints_neuronswap/mid2/checkpoint-200_metadata.json \
    --output-dir results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis \
    --num-points 61
```

### Step 4: Create Visualization

```bash
python scripts/plotting/plot_layer_distance_animation.py \
    --data results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_along_curve.npz \
    --output results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_evolution.gif \
    --fps 10 \
    --metric normalized_l2 \
    --heatmap
```

---

## File Structure

```
Mode-Connectivity/
├── scripts/
│   ├── analysis/
│   │   ├── swap_neurons.py              # Create swapped networks
│   │   └── analyze_neuron_swap_distances.py  # Analyze layer distances
│   ├── plotting/
│   │   └── plot_layer_distance_animation.py  # Create GIF visualization
│   ├── workflows/
│   │   └── setup_neuronswap_experiments.sh   # Setup script
│   └── slurm/
│       ├── submit_neuronswap_curve_early2.sh  # Train early swap curve
│       ├── submit_neuronswap_curve_mid2.sh    # Train mid swap curve
│       ├── submit_neuronswap_curve_late2.sh   # Train late swap curve
│       └── submit_neuronswap_analysis.sh      # Run analysis
├── configs/garipov/
│   ├── vgg16_curve_neuronswap_early2_reg.yaml
│   ├── vgg16_curve_neuronswap_mid2_reg.yaml
│   └── vgg16_curve_neuronswap_late2_reg.yaml
└── results/vgg16/cifar10/
    ├── endpoints_neuronswap/
    │   ├── early2/
    │   │   ├── checkpoint-200.pt
    │   │   └── checkpoint-200_metadata.json
    │   ├── mid2/
    │   │   ├── checkpoint-200.pt
    │   │   └── checkpoint-200_metadata.json
    │   └── late2/
    │       ├── checkpoint-200.pt
    │       └── checkpoint-200_metadata.json
    └── curve_neuronswap_[depth]_reg/
        ├── checkpoints/
        │   └── checkpoint-200.pt
        └── analysis/
            ├── layer_distances_along_curve.npz
            ├── analysis_summary.json
            ├── layer_distances_evolution.gif
            └── layer_distances_evolution_heatmap.png
```

---

## Key Scripts

### 1. `swap_neurons.py`

Creates a functionally equivalent network by swapping neurons.

**Usage:**
```bash
python scripts/analysis/swap_neurons.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --output PATH_TO_SAVE \
    --layer-depth {early,mid,late} \
    --num-swaps NUM_PAIRS \
    --seed RANDOM_SEED \
    [--verify]  # Full dataset verification (slower)
```

**Parameters:**
- `--layer-depth`: Which layer to swap
  - `early`: Block 0, Layer 0 (64 filters)
  - `mid`: Block 2, Layer 1 (256 filters)
  - `late`: Block 4, Layer 1 (512 filters)
- `--num-swaps`: Number of neuron pairs to swap (default: 1 = 2 neurons)
- `--seed`: Random seed for reproducible neuron selection
- `--verify`: Verify functional equivalence on full test set

**Output:**
- Swapped checkpoint (`.pt` file)
- Metadata JSON with swap details

### 2. `analyze_neuron_swap_distances.py`

Evaluates layer-wise distances along the curve.

**Usage:**
```bash
python scripts/analysis/analyze_neuron_swap_distances.py \
    --curve-checkpoint PATH_TO_CURVE \
    --original-checkpoint PATH_TO_ORIGINAL \
    --swap-metadata PATH_TO_METADATA \
    --output-dir OUTPUT_DIR \
    --num-points 61
```

**Output:**
- `layer_distances_along_curve.npz`: NumPy arrays with distance data
- `analysis_summary.json`: Human-readable summary

### 3. `plot_layer_distance_animation.py`

Creates animated visualization.

**Usage:**
```bash
python scripts/plotting/plot_layer_distance_animation.py \
    --data PATH_TO_NPZ \
    --output PATH_TO_GIF \
    --fps 10 \
    --metric {normalized_l2,relative,raw_l2} \
    [--heatmap]  # Also create static heatmap
```

**Output:**
- Animated GIF (61 frames)
- Optional: Static heatmap PNG

---

## Distance Metrics

### Normalized L2
$$\text{Normalized L2} = \frac{\|W_t - W_0\|_2}{\sqrt{n_{params}}}$$

- **Use case**: Comparing distances across layers with different sizes
- **Interpretation**: Average per-parameter change
- **Best for**: Cross-layer comparison

### Relative Distance
$$\text{Relative} = \frac{\|W_t - W_0\|_2}{\|W_0\|_2}$$

- **Use case**: Percentage change relative to original weights
- **Interpretation**: Proportional change
- **Best for**: Understanding magnitude of change

### Raw L2
$$\text{Raw L2} = \|W_t - W_0\|_2$$

- **Use case**: Absolute distance in weight space
- **Interpretation**: Euclidean distance
- **Note**: Biased toward large layers

---

## Expected Results

### Hypothesis A: Local Correction
**If corrections are local:**
- Swapped layer shows **high distance** from original
- Other layers show **minimal distance**
- Animation: Red bar (swapped layer) is tall, others are short

**Interpretation:**
- Weight space has strong local structure
- Permutations create "tunnels" in loss landscape
- Mode connectivity is geometrically simple

### Hypothesis B: Global Adjustment
**If corrections are global:**
- **All layers** show significant distance from original
- Swapped layer may not be the most changed
- Animation: Many bars grow tall

**Interpretation:**
- Weight space has non-local, entangled structure
- Permutations create global effects
- Mode connectivity requires coordinated adjustment

### Layer Depth Effects
**Early layers (low-level features):**
- High fan-out → swaps may require global adjustment
- Many downstream neurons depend on these features

**Late layers (high-level features):**
- Low fan-out → swaps may be more local
- Fewer downstream dependencies

---

## Troubleshooting

### Verification fails
If `swap_neurons.py` reports verification failure:
1. Check if using correct checkpoint
2. Ensure model architecture matches
3. Try with `--no-verify` to skip (not recommended)

### Analysis script errors
If `analyze_neuron_swap_distances.py` fails:
1. Ensure curve training completed successfully
2. Check that all checkpoint paths are correct
3. Verify metadata JSON is readable

### GIF creation fails
If `plot_layer_distance_animation.py` fails:
1. Install Pillow: `pip install Pillow`
2. Check that NPZ file exists and is valid
3. Try with fewer frames: `--num-points 31`

---

## Advanced Usage

### Custom Layer Selection

Instead of using `--layer-depth`, specify exact layer:

```python
# In swap_neurons.py, modify get_layer_by_depth() to return specific layer
# Or implement --layer-name parameter for manual selection
```

### Multiple Swap Amounts

Test different numbers of swaps:

```bash
# 2 neurons (1 pair)
python scripts/analysis/swap_neurons.py ... --num-swaps 1

# 10 neurons (5 pairs)
python scripts/analysis/swap_neurons.py ... --num-swaps 5

# 20 neurons (10 pairs)
python scripts/analysis/swap_neurons.py ... --num-swaps 10
```

### Different Distance Metrics

Compare metrics side-by-side:

```bash
# Create animations for all metrics
for metric in normalized_l2 relative raw_l2; do
    python scripts/plotting/plot_layer_distance_animation.py \
        --data data.npz \
        --output "animation_${metric}.gif" \
        --metric $metric
done
```

---

## Related Experiments

- **Mirror experiments** (`curve_seed0-mirror_reg`): Full network permutation
- **Seed experiments** (`curve_seed0-seed1_reg`): Different random initializations

Compare:
- **Neuron swap**: 2 neurons swapped → minimal perturbation
- **Mirror**: All neurons swapped → maximal permutation
- **Different seeds**: No permutation → random initialization

---

## Citations

If you use this experimental setup, consider citing:

- **Mode Connectivity**: Garipov et al. (2018) "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs"
- **Permutation Symmetry**: Entezari et al. (2021) "The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks"

---

## Questions?

For questions or issues, open an issue at:
https://github.com/anthropics/mode-connectivity/issues

Or contact: mlodzinski@student.tudelft.nl

---

## License

Same as main project (MIT License)
