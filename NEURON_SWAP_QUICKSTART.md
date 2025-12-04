# Neuron Swap Experiments - Quick Start Guide

## What This Tests

**Research Question:** When we swap just 2 neurons in a network, does the connectivity path correct:
- **Locally**: Only the 2 swapped neurons adjust?
- **Globally**: All weights readjust throughout the network?

This reveals whether neural network weight spaces have local or entangled structure.

---

## Complete Pipeline (4 Steps)

### Step 1: Create Swapped Endpoints (Local, ~5 minutes)

```bash
bash scripts/workflows/setup_neuronswap_experiments.sh
```

This creates 3 swapped versions of seed0:
- `early2`: 2 neurons swapped in Block 0 (early features)
- `mid2`: 2 neurons swapped in Block 2 (mid features)
- `late2`: 2 neurons swapped in Block 4 (late features)

Each is functionally identical (same accuracy) but at different point in weight space.

**Output:**
- `results/vgg16/cifar10/endpoints_neuronswap/{early2,mid2,late2}/checkpoint-200.pt`
- Metadata JSON files showing which neurons were swapped

---

### Step 2: Train Bezier Curves (Cluster, ~2h each)

```bash
# Submit all three jobs
sbatch scripts/slurm/submit_neuronswap_curve_early2.sh
sbatch scripts/slurm/submit_neuronswap_curve_mid2.sh
sbatch scripts/slurm/submit_neuronswap_curve_late2.sh
```

This trains curves connecting original ↔ swapped for each depth.

**Output:**
- `results/vgg16/cifar10/curve_neuronswap_{early2,mid2,late2}_reg/checkpoints/checkpoint-200.pt`

**Monitor:** Check SLURM output files or WandB (project: thesis-mode-connectivity)

---

### Step 3: Analyze Layer Distances (Cluster, ~10 minutes)

**Edit the script first:**
```bash
# Open scripts/slurm/submit_neuronswap_analysis.sh
# Change line: EXPERIMENT="mid2"
# Options: early2, mid2, late2
```

Then submit:
```bash
sbatch scripts/slurm/submit_neuronswap_analysis.sh
```

Or run locally (if you have the curve checkpoint):
```bash
# Example for mid-layer experiment
EXPERIMENT="mid2"

python scripts/analysis/analyze_neuron_swap_distances.py \
    --curve-checkpoint "results/vgg16/cifar10/curve_neuronswap_${EXPERIMENT}_reg/checkpoints/checkpoint-200.pt" \
    --original-checkpoint "results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt" \
    --swap-metadata "results/vgg16/cifar10/endpoints_neuronswap/${EXPERIMENT}/checkpoint-200_metadata.json" \
    --output-dir "results/vgg16/cifar10/curve_neuronswap_${EXPERIMENT}_reg/analysis" \
    --num-points 61
```

**Output:**
- `layer_distances_along_curve.npz` - Distance data for all layers
- `analysis_summary.json` - Readable summary

---

### Step 4: Create Animated Visualization (Local/Cluster, ~2 minutes)

```bash
# Example for mid-layer experiment
EXPERIMENT="mid2"

python scripts/plotting/plot_layer_distance_animation.py \
    --data "results/vgg16/cifar10/curve_neuronswap_${EXPERIMENT}_reg/analysis/layer_distances_along_curve.npz" \
    --output "results/vgg16/cifar10/curve_neuronswap_${EXPERIMENT}_reg/analysis/layer_distances_evolution.gif" \
    --fps 10 \
    --metric normalized_l2 \
    --heatmap
```

**Output:**
- `layer_distances_evolution.gif` - Animated bar chart (61 frames)
- `layer_distances_evolution_heatmap.png` - Static heatmap

---

## Interpreting Results

### Reading the Animation

**X-axis:** Layer index (0 = early conv, higher = later layers)
**Y-axis:** Normalized L2 distance from original network
**Red bar:** The layer where neurons were swapped
**Progress bar:** Position along curve (t = 0 → 1)

### Hypothesis A: Local Correction ✓

**What to look for:**
- RED bar (swapped layer) is TALL
- Other bars stay SHORT
- Only swapped layer changes significantly

**Interpretation:**
- Weight space has local structure
- Permutations are easy to "undo"
- Mode connectivity is geometrically simple

### Hypothesis B: Global Adjustment ✓

**What to look for:**
- MANY bars grow tall (not just red one)
- Changes spread across layers
- Swapped layer may not be the most changed

**Interpretation:**
- Weight space has entangled structure
- Permutations create global effects
- Requires coordinated adjustment across layers

---

## Common Issues

### Issue 1: "Checkpoint not found"
**Solution:** Make sure Step 1 completed successfully. Check that:
```bash
ls results/vgg16/cifar10/endpoints/checkpoints/seed0/checkpoint-200.pt
```

### Issue 2: Curve training doesn't start
**Solution:** Check SLURM output files:
```bash
tail slurm_neuronswap_curve_mid2_*.err
```

### Issue 3: Analysis crashes with "Layer not found"
**Possible cause:** Curve training didn't complete (200 epochs)
**Solution:** Check curve checkpoint epoch:
```python
import torch
ckpt = torch.load('path/to/curve/checkpoint-200.pt', map_location='cpu')
print(ckpt.get('epoch', 'No epoch info'))
```

### Issue 4: GIF is corrupted or doesn't play
**Solution:** Install/update Pillow:
```bash
pip install --upgrade Pillow
```

---

## Quick Checklist

- [ ] Step 1: Created swapped endpoints
- [ ] Step 2: Trained curves (early, mid, late)
- [ ] Step 3: Analyzed distances (early, mid, late)
- [ ] Step 4: Created visualizations (early, mid, late)
- [ ] Compared results across depths
- [ ] Interpreted findings (local vs global)

---

## Time Estimates

| Step | Time | Where |
|------|------|-------|
| Create endpoints | 5 min | Local |
| Train 1 curve | 2 hours | Cluster (GPU) |
| Train all 3 curves | 2 hours | Cluster (parallel) |
| Analyze 1 curve | 10 min | Cluster/Local |
| Create visualization | 2 min | Local |
| **Total (parallel)** | **~2.5 hours** | |

---

## Files You'll Create

Total files per experiment (early, mid, late):

**Endpoints:**
- checkpoint-200.pt (swapped network)
- checkpoint-200_metadata.json (swap details)

**Curves:**
- checkpoint-200.pt (trained curve)
- command.sh (training command)

**Analysis:**
- layer_distances_along_curve.npz (distance data)
- analysis_summary.json (summary)
- layer_distances_evolution.gif (animation)
- layer_distances_evolution_heatmap.png (heatmap)

**Total storage:** ~500MB per experiment

---

## Next Steps After Completion

1. **Compare across depths:**
   - Are early/mid/late swaps corrected differently?
   - Which depth shows most local correction?

2. **Test different swap amounts:**
   - Modify `--num-swaps` in setup script
   - Compare: 1 pair (2 neurons) vs 5 pairs (10 neurons) vs 10 pairs (20 neurons)

3. **Compare to full mirror:**
   - How do 2-neuron swaps compare to all-neuron swap?
   - Check existing `curve_seed0-mirror_reg` results

4. **Write up findings:**
   - Local vs global correction?
   - Layer depth effects?
   - Implications for loss landscape structure?

---

## For More Details

See [NEURON_SWAP_EXPERIMENTS.md](NEURON_SWAP_EXPERIMENTS.md) for:
- Detailed documentation
- Advanced usage
- Troubleshooting guide
- Mathematical formulas
- Related experiments

---

## Questions?

Contact: mlodzinski@student.tudelft.nl
GitHub Issues: https://github.com/anthropics/mode-connectivity/issues
