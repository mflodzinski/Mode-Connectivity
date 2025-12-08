# Neuron Swap Experiment Results - Quick Reference

## 📊 Viewing Results

### **1. Check L2 Distances Between Endpoints**

Shows how far apart the original and swapped networks are in weight space:

```bash
# View all endpoint distances
cat results/vgg16/cifar10/curve_neuronswap_early2_reg/checkpoints/endpoint_l2_distance.txt
cat results/vgg16/cifar10/curve_neuronswap_mid2_reg/checkpoints/endpoint_l2_distance.txt
cat results/vgg16/cifar10/curve_neuronswap_late2_reg/checkpoints/endpoint_l2_distance.txt

# Or all at once
cat results/vgg16/cifar10/curve_neuronswap_*/checkpoints/endpoint_l2_distance.txt
```

**Expected:** Very small normalized L2 (~0.003) - confirming minimal perturbation

---

### **2. View Animated Layer Distance Evolution**

Shows which layers change along the connectivity path (answers: local vs global?):

```bash
# Open all animations
open results/vgg16/cifar10/curve_neuronswap_early2_reg/analysis/layer_distances_evolution.gif
open results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_evolution.gif
open results/vgg16/cifar10/curve_neuronswap_late2_reg/analysis/layer_distances_evolution.gif

# Or all at once
open results/vgg16/cifar10/curve_neuronswap_*/analysis/layer_distances_evolution.gif
```

**What to look for:**
- **Local correction:** Only swapped layer + adjacent layer show tall bars
- **Global correction:** Many layers show significant bars

---

### **3. Compare Static Heatmaps**

Shows all layers × all timepoints at once:

```bash
# Open all heatmaps
open results/vgg16/cifar10/curve_neuronswap_early2_reg/analysis/layer_distances_evolution_heatmap.png
open results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_evolution_heatmap.png
open results/vgg16/cifar10/curve_neuronswap_late2_reg/analysis/layer_distances_evolution_heatmap.png

# Or all at once
open results/vgg16/cifar10/curve_neuronswap_*/analysis/layer_distances_evolution_heatmap.png
```

**What to look for:**
- Horizontal bands = layers that change
- Should see 1-2 bright bands (swapped + adjacent), rest dark

---

### **4. Check Analysis Summaries**

JSON files with detailed statistics:

```bash
# View summary
cat results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/analysis_summary.json

# Or use jq for pretty printing
cat results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/analysis_summary.json | jq
```

---

### **5. Inspect Raw Data**

NPZ files contain the actual distance arrays:

```python
import numpy as np

# Load data
data = np.load('results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_along_curve.npz')

# Available arrays
print(data.files)
# ['layer_names', 'layer_types', 'layer_params', 't_values',
#  'normalized_l2', 'relative', 'raw_l2', 'swapped_block', 'swapped_layer']

# Access data
layer_names = data['layer_names']
distances = data['normalized_l2']  # Shape: (61 timepoints, num_layers)
```

---

## 🔍 Key Questions to Answer

### **Q1: Are corrections local or global?**

**Check:** Animations and heatmaps

- ✅ **Local**: 1-2 layers change (swapped + adjacent)
- ❌ **Global**: Many layers change

### **Q2: Does layer depth matter?**

**Compare:** Early vs Mid vs Late animations side-by-side

- Do different depths show different patterns?
- Which shows most localized correction?

### **Q3: How does 2-neuron swap compare to full permutation?**

**Compare L2 distances:**

```bash
# 2-neuron swap (should be tiny)
cat results/vgg16/cifar10/curve_neuronswap_mid2_reg/checkpoints/endpoint_l2_distance.txt

# Full mirror (should be large) - if you have it
cat results/vgg16/cifar10/curve_seed0-mirror_reg/checkpoints/endpoint_l2_distance.txt
```

**Expected:** Mirror is ~100-1000x larger

---

## 📈 Creating Comparison Plots

### **Side-by-side comparison script:**

```python
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, depth in enumerate(['early2', 'mid2', 'late2']):
    data = np.load(f'results/vgg16/cifar10/curve_neuronswap_{depth}_reg/analysis/layer_distances_along_curve.npz')

    # Plot heatmap
    im = axes[i].imshow(data['normalized_l2'].T, aspect='auto', cmap='viridis')
    axes[i].set_title(f'{depth.upper()} Layer Swap')
    axes[i].set_xlabel('Position along curve (t)')
    axes[i].set_ylabel('Layer Index')

    # Mark swapped layer
    swapped = data['swapped_layer']
    axes[i].axhline(y=swapped, color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig('comparison_all_depths.png', dpi=300)
plt.show()
```

---

## 🎯 Quick Interpretation Guide

### **Expected Pattern (Local Correction):**

```
Animation shows:
  Layer 0-10:   Flat (near zero)
  Layer X:      Tall bars ← Swapped layer
  Layer X+1:    Medium bars ← Adjacent layer (chaining)
  Layer X+2:    Flat (near zero)
  ...
  Layer 15:     Flat (near zero)
```

### **Unexpected Pattern (Global Correction):**

```
Animation shows:
  Layer 0:      Medium bars
  Layer 5:      Medium bars
  Layer X:      Tall bars ← Swapped layer
  Layer X+5:    Medium bars
  Layer 15:     Medium bars
```

If you see the unexpected pattern, it means the network requires global weight adjustments!

---

## 📁 File Structure Reference

```
results/vgg16/cifar10/curve_neuronswap_<depth>_reg/
├── checkpoints/
│   ├── checkpoint-30.pt                    # Trained curve
│   └── endpoint_l2_distance.txt            # L2 between endpoints
└── analysis/
    ├── layer_distances_along_curve.npz     # Raw data
    ├── analysis_summary.json               # Summary stats
    ├── layer_distances_evolution.gif       # Animation (KEY!)
    └── layer_distances_evolution_heatmap.png # Static view
```

---

## 💡 Tips

**If animations don't show clearly:**
- Try the heatmaps instead (static, easier to compare)
- Check `analysis_summary.json` for swapped layer info

**If you see all zeros:**
- Check that curve training completed (30 epochs)
- Verify checkpoint paths are correct

**To generate new visualizations:**
```bash
# Re-run visualization with different settings
python scripts/plotting/plot_layer_distance_animation.py \
    --data results/vgg16/cifar10/curve_neuronswap_mid2_reg/analysis/layer_distances_along_curve.npz \
    --output custom_animation.gif \
    --fps 5 \
    --metric relative  # Try different metrics
```

---

## 🚀 One-liner: View Everything

```bash
# Open all visualizations at once
open results/vgg16/cifar10/curve_neuronswap_*/analysis/*.{gif,png}

# View all L2 distances
cat results/vgg16/cifar10/curve_neuronswap_*/checkpoints/endpoint_l2_distance.txt
```

---

## ✅ Success Criteria

Your experiment is successful if you can answer:

- ✅ What is the L2 distance for 2-neuron swaps? (~0.001-0.01 expected)
- ✅ Which layers change along the path? (Only swapped + adjacent expected)
- ✅ Does layer depth affect the pattern? (Compare early/mid/late)
- ✅ Is correction local or global? (Local expected)

**These results directly test your hypothesis about weight space structure!**
