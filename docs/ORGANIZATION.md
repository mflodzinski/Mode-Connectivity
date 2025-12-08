# Project Organization

## 📂 Directory Structure

The project has been organized into logical categories for better maintainability.

---

## 🎯 SLURM Scripts (`scripts/slurm/`)

```
scripts/slurm/
├── endpoints/          Training endpoint models
│   ├── submit_garipov.sh
│   └── submit_garipov_regularized_endpoints.sh
│
├── curves/            Training connectivity curves
│   ├── submit_garipov_curve_seed0-seed1_reg.sh
│   ├── submit_garipov_curve_seed0-seed1_noreg.sh
│   ├── submit_garipov_curve_seed0-mirror_reg.sh
│   └── submit_garipov_curve_seed0-mirror_noreg.sh
│
├── evaluation/        Evaluating trained curves
│   ├── submit_garipov_eval_seed0-seed1_reg.sh
│   ├── submit_garipov_eval_seed0-seed1_noreg.sh
│   ├── submit_garipov_eval_seed0-mirror_reg.sh
│   ├── submit_garipov_eval_seed0-mirror_noreg.sh
│   ├── submit_garipov_eval_l2_test.sh
│   └── submit_prediction_changes_eval.sh
│
├── neuronswap/        Neuron swap experiments
│   ├── submit_neuronswap_curve_early2.sh
│   ├── submit_neuronswap_curve_mid2.sh
│   ├── submit_neuronswap_curve_late2.sh
│   └── submit_neuronswap_analysis.sh
│
└── pipelines/         Full multi-step pipelines
    └── submit_garipov_full_pipeline.sh
```

---

## ⚙️ Configuration Files (`configs/garipov/`)

```
configs/garipov/
├── endpoints/         Endpoint training configs
│   ├── vgg16_endpoints.yaml
│   └── vgg16_regularized_endpoints.yaml
│
├── curves/           Curve training configs
│   ├── vgg16_curve_seed0-seed1_reg.yaml
│   ├── vgg16_curve_seed0-seed1_noreg.yaml
│   ├── vgg16_curve_seed0-mirror_reg.yaml
│   ├── vgg16_curve_seed0-mirror_noreg.yaml
│   └── vgg16_curve_l2_test.yaml
│
└── neuronswap/       Neuron swap configs
    ├── vgg16_curve_neuronswap_early2_reg.yaml
    ├── vgg16_curve_neuronswap_mid2_reg.yaml
    └── vgg16_curve_neuronswap_late2_reg.yaml
```

---

## 🚀 Quick Reference

### Training Endpoints

```bash
# Train initial models (seed0, seed1)
sbatch scripts/slurm/endpoints/submit_garipov.sh

# Train regularized endpoints
sbatch scripts/slurm/endpoints/submit_garipov_regularized_endpoints.sh
```

### Training Curves

```bash
# Regular experiments
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-seed1_reg.sh
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-mirror_reg.sh

# Without regularization
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-seed1_noreg.sh
sbatch scripts/slurm/curves/submit_garipov_curve_seed0-mirror_noreg.sh
```

### Neuron Swap Experiments

```bash
# Train curves
sbatch scripts/slurm/neuronswap/submit_neuronswap_curve_early2.sh
sbatch scripts/slurm/neuronswap/submit_neuronswap_curve_mid2.sh
sbatch scripts/slurm/neuronswap/submit_neuronswap_curve_late2.sh

# Analyze results
sbatch scripts/slurm/neuronswap/submit_neuronswap_analysis.sh
```

### Evaluation

```bash
# Evaluate specific curves
sbatch scripts/slurm/evaluation/submit_garipov_eval_seed0-seed1_reg.sh
sbatch scripts/slurm/evaluation/submit_garipov_eval_seed0-mirror_reg.sh

# Prediction analysis
sbatch scripts/slurm/evaluation/submit_prediction_changes_eval.sh
```

### Full Pipeline

```bash
# Run complete workflow (train + evaluate)
sbatch scripts/slurm/pipelines/submit_garipov_full_pipeline.sh
```

---

## 📝 Config Path Updates

### Using Hydra Configs

The training scripts use Hydra with the new organized structure:

**For curve training:**
```python
# Old (deprecated)
@hydra.main(config_path="../../configs/garipov", config_name="vgg16_curve_seed0-seed1_reg")

# New
@hydra.main(config_path="../../configs/garipov/curves", config_name="vgg16_curve_seed0-seed1_reg")
```

**For neuron swap:**
```python
@hydra.main(config_path="../../configs/garipov/neuronswap", config_name="vgg16_curve_neuronswap_mid2_reg")
```

**For endpoints:**
```python
@hydra.main(config_path="../../configs/garipov/endpoints", config_name="vgg16_endpoints")
```

---

## 🔄 Migration Notes

### Files Moved

**SLURM Scripts:**
- Endpoint training → `scripts/slurm/endpoints/`
- Curve training → `scripts/slurm/curves/`
- Evaluation → `scripts/slurm/evaluation/`
- Neuron swap → `scripts/slurm/neuronswap/`
- Pipelines → `scripts/slurm/pipelines/`

**Configs:**
- Endpoint configs → `configs/garipov/endpoints/`
- Curve configs → `configs/garipov/curves/`
- Neuron swap configs → `configs/garipov/neuronswap/`

### Backward Compatibility

⚠️ **Old paths no longer work!** Update any scripts or documentation that reference:
- `scripts/slurm/submit_*.sh` → Use new categorized paths
- `configs/garipov/vgg16_*.yaml` → Use new subdirectory paths

---

## 🎯 Category Descriptions

### **Endpoints**
Initial model training with different random seeds or configurations.

**Purpose:** Create the "modes" that will later be connected

**Files:**
- Training scripts for seed0, seed1
- Configs for standard and regularized training

---

### **Curves**
Training Bezier curves to connect different endpoint pairs.

**Purpose:** Find low-loss paths between modes

**Experiments:**
- `seed0-seed1`: Different random initializations
- `seed0-mirror`: Full neuron permutation
- `*_reg`: With L2 regularization (wd=5e-4)
- `*_noreg`: Without regularization (wd=0.0)

---

### **Evaluation**
Evaluate trained curves and analyze connectivity properties.

**Purpose:** Measure loss barriers, prediction stability

**Analyses:**
- Curve evaluation (loss/error at 61 points)
- Linear interpolation baseline
- Prediction change analysis

---

### **Neuron Swap**
Minimal perturbation experiments - swap just 2 neurons.

**Purpose:** Test local vs global connectivity corrections

**Experiments:**
- `early2`: 2 neurons in early layer (Block 0)
- `mid2`: 2 neurons in mid layer (Block 2)
- `late2`: 2 neurons in late layer (Block 4)

---

### **Pipelines**
Complete multi-step workflows combining training and evaluation.

**Purpose:** Reproducible end-to-end experiments

**Workflows:**
- Full pipeline: train endpoints → train curve → evaluate curve → evaluate linear

---

## 🔍 Finding Files

### By Experiment Type

```bash
# All endpoint-related files
ls scripts/slurm/endpoints/
ls configs/garipov/endpoints/

# All neuron swap files
ls scripts/slurm/neuronswap/
ls configs/garipov/neuronswap/

# All evaluation scripts
ls scripts/slurm/evaluation/
```

### By Regularization

```bash
# With regularization
ls scripts/slurm/curves/*_reg.sh

# Without regularization
ls scripts/slurm/curves/*_noreg.sh
```

### By Endpoint Pair

```bash
# seed0-seed1 experiments
ls scripts/slurm/curves/*seed0-seed1*.sh
ls scripts/slurm/evaluation/*seed0-seed1*.sh

# seed0-mirror experiments
ls scripts/slurm/curves/*seed0-mirror*.sh
ls scripts/slurm/evaluation/*seed0-mirror*.sh
```

---

## ✅ Benefits of New Organization

1. **Clarity:** Immediately understand what each script does
2. **Maintainability:** Easier to find and update related files
3. **Scalability:** Simple to add new experiment types
4. **Documentation:** Self-documenting directory structure
5. **Collaboration:** Clear organization for other researchers

---

## 📚 Related Documentation

- **[NEURON_SWAP_EXPERIMENTS.md](NEURON_SWAP_EXPERIMENTS.md)** - Neuron swap details
- **[NEURON_SWAP_QUICKSTART.md](NEURON_SWAP_QUICKSTART.md)** - Quick start guide
- **[NEURON_SWAP_RESULTS.md](NEURON_SWAP_RESULTS.md)** - Results viewing guide
- **[L2_DISTANCE_TRACKING.md](L2_DISTANCE_TRACKING.md)** - L2 distance documentation
- **[README.md](../README.md)** - Main project documentation
