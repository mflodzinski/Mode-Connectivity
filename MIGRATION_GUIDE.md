# Migration Guide - Organized Directory Structure

## ✅ Changes Made

All SLURM scripts and configs have been organized into logical subdirectories.

---

## 📋 Quick Reference

### **Old paths → New paths**

#### SLURM Scripts

```bash
# Endpoints
scripts/slurm/submit_garipov.sh
  → scripts/slurm/endpoints/submit_garipov.sh

# Curves
scripts/slurm/submit_garipov_curve_seed0-seed1_reg.sh
  → scripts/slurm/curves/submit_garipov_curve_seed0-seed1_reg.sh

# Evaluation
scripts/slurm/submit_garipov_eval_seed0-seed1_reg.sh
  → scripts/slurm/evaluation/submit_garipov_eval_seed0-seed1_reg.sh

# Neuron swap
scripts/slurm/submit_neuronswap_curve_mid2.sh
  → scripts/slurm/neuronswap/submit_neuronswap_curve_mid2.sh

# Pipelines
scripts/slurm/submit_garipov_full_pipeline.sh
  → scripts/slurm/pipelines/submit_garipov_full_pipeline.sh
```

#### Configs

```bash
# Endpoints
configs/garipov/vgg16_endpoints.yaml
  → configs/garipov/endpoints/vgg16_endpoints.yaml

# Curves
configs/garipov/vgg16_curve_seed0-seed1_reg.yaml
  → configs/garipov/curves/vgg16_curve_seed0-seed1_reg.yaml

# Neuron swap
configs/garipov/vgg16_curve_neuronswap_mid2_reg.yaml
  → configs/garipov/neuronswap/vgg16_curve_neuronswap_mid2_reg.yaml
```

---

## 🔧 Updating Your Commands

### **Before:**
```bash
sbatch scripts/slurm/submit_garipov_curve_seed0-seed1_reg.sh
```

### **After:**
```bash
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-seed1_reg.sh
```

---

## 🐍 Updating Python Scripts

### **If you have custom scripts using Hydra:**

**Before:**
```python
@hydra.main(
    config_path="../../configs/garipov",
    config_name="vgg16_curve_seed0-seed1_reg"
)
```

**After:**
```python
@hydra.main(
    config_path="../../configs/garipov/curves",  # ← Add subdirectory
    config_name="vgg16_curve_seed0-seed1_reg"
)
```

**Or use search_path (recommended):**
```python
@hydra.main(
    config_path="../../configs/garipov",  # Search from root
    config_name="curves/vgg16_curve_seed0-seed1_reg"  # ← Include subdirectory in name
)
```

---

## 📂 Complete Path Mapping

### Endpoints

| Old | New |
|-----|-----|
| `scripts/slurm/submit_garipov.sh` | `scripts/slurm/endpoints/submit_garipov.sh` |
| `scripts/slurm/submit_garipov_regularized_endpoints.sh` | `scripts/slurm/endpoints/submit_garipov_regularized_endpoints.sh` |
| `configs/garipov/vgg16_endpoints.yaml` | `configs/garipov/endpoints/vgg16_endpoints.yaml` |
| `configs/garipov/vgg16_regularized_endpoints.yaml` | `configs/garipov/endpoints/vgg16_regularized_endpoints.yaml` |

### Curves

| Old | New |
|-----|-----|
| `scripts/slurm/submit_garipov_curve_seed0-seed1_reg.sh` | `scripts/slurm/curves/submit_garipov_curve_seed0-seed1_reg.sh` |
| `scripts/slurm/submit_garipov_curve_seed0-seed1_noreg.sh` | `scripts/slurm/curves/submit_garipov_curve_seed0-seed1_noreg.sh` |
| `scripts/slurm/submit_garipov_curve_seed0-mirror_reg.sh` | `scripts/slurm/curves/submit_garipov_curve_seed0-mirror_reg.sh` |
| `scripts/slurm/submit_garipov_curve_seed0-mirror_noreg.sh` | `scripts/slurm/curves/submit_garipov_curve_seed0-mirror_noreg.sh` |
| `configs/garipov/vgg16_curve_*.yaml` | `configs/garipov/curves/vgg16_curve_*.yaml` |

### Evaluation

| Old | New |
|-----|-----|
| `scripts/slurm/submit_garipov_eval_*.sh` | `scripts/slurm/evaluation/submit_garipov_eval_*.sh` |
| `scripts/slurm/submit_prediction_changes_eval.sh` | `scripts/slurm/evaluation/submit_prediction_changes_eval.sh` |

### Neuron Swap

| Old | New |
|-----|-----|
| `scripts/slurm/submit_neuronswap_*.sh` | `scripts/slurm/neuronswap/submit_neuronswap_*.sh` |
| `configs/garipov/vgg16_curve_neuronswap_*.yaml` | `configs/garipov/neuronswap/vgg16_curve_neuronswap_*.yaml` |

### Pipelines

| Old | New |
|-----|-----|
| `scripts/slurm/submit_garipov_full_pipeline.sh` | `scripts/slurm/pipelines/submit_garipov_full_pipeline.sh` |

---

## ✅ No Action Needed

The main training scripts ([run_garipov_curve.py](scripts/train/run_garipov_curve.py), [run_garipov_endpoints.py](scripts/train/run_garipov_endpoints.py)) **do not need updates** because they use the `--config-name` parameter, which Hydra resolves correctly with the new structure.

---

## 🔍 Finding Files

Use these commands to quickly locate files:

```bash
# Find all SLURM scripts
find scripts/slurm -name "*.sh"

# Find all configs
find configs/garipov -name "*.yaml"

# Find scripts for specific experiment
ls scripts/slurm/neuronswap/
ls configs/garipov/neuronswap/

# Find by regularization
ls scripts/slurm/curves/*_reg.sh
ls scripts/slurm/curves/*_noreg.sh
```

---

## 📚 Documentation Updated

- **[ORGANIZATION.md](ORGANIZATION.md)** - Complete structure reference
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - This file
- **[NEURON_SWAP_RESULTS.md](NEURON_SWAP_RESULTS.md)** - No changes needed
- **[NEURON_SWAP_QUICKSTART.md](NEURON_SWAP_QUICKSTART.md)** - No changes needed (uses new paths)

---

## ⚠️ Breaking Changes

**These will NOT work anymore:**

```bash
# ❌ Old (broken)
sbatch scripts/slurm/submit_garipov_curve_seed0-seed1_reg.sh
python scripts/train/run_garipov_curve.py --config-name vgg16_curve_seed0-seed1_reg

# ✅ New (correct)
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-seed1_reg.sh
python scripts/train/run_garipov_curve.py --config-name curves/vgg16_curve_seed0-seed1_reg
```

Actually, the Python script should still work as Hydra searches subdirectories, but it's clearer to be explicit.

---

## 💡 Tips

1. **Use tab completion**: Type `sbatch scripts/slurm/` then press TAB to see categories
2. **Bookmark ORGANIZATION.md**: Quick reference for all paths
3. **Use descriptive names**: The new structure is self-documenting
4. **Grep for files**: `grep -r "config_name" scripts/` to find all config references

---

## 🎯 Benefits

- ✅ Clearer organization (find files by category)
- ✅ Easier to onboard new researchers
- ✅ Scalable (easy to add new experiment types)
- ✅ Self-documenting directory structure
- ✅ Less clutter in main directories

---

## 📞 Questions?

See **[ORGANIZATION.md](ORGANIZATION.md)** for the complete directory structure and usage examples.
