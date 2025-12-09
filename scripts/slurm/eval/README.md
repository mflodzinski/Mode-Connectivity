# Evaluation Scripts for SLURM Cluster

## Individual Evaluations

### Polygon Chain (seed0-mirror)
```bash
sbatch scripts/slurm/eval/submit_eval_polygon_seed0-mirror.sh
```

### Symmetry Plane (seed0-mirror)
```bash
sbatch scripts/slurm/eval/submit_eval_symplane_seed0-mirror.sh
```

### Symmetry Plane (seed0-seed1)
```bash
sbatch scripts/slurm/eval/submit_eval_symplane_seed0-seed1.sh
```

## Submit All at Once

```bash
bash scripts/slurm/eval/submit_all_evals.sh
```

## Check Job Status

```bash
squeue -u $USER
```

## View Output

```bash
# Check SLURM output logs
tail -f slurm_eval_polygon_seed0-mirror_*.out
tail -f slurm_eval_symplane_seed0-mirror_*.out
tail -f slurm_eval_symplane_seed0-seed1_*.out

# Check evaluation results
ls results/vgg16/cifar10/polygon_seed0-mirror/evaluations/
ls results/vgg16/cifar10/symmetry_plane_seed0-mirror/evaluations/
ls results/vgg16/cifar10/symmetry_plane_seed0-seed1/evaluations/
```

## Expected Output

Each evaluation creates a `curve.npz` file containing:
- `ts`: t values (0 to 1)
- `tr_loss`, `tr_acc`, `tr_err`: Training metrics
- `te_loss`, `te_acc`, `te_err`: Test metrics

## Notes

- Each evaluation takes ~30-60 minutes on A40 GPU
- Uses 61 evaluation points along the curve
- Results saved to `results/vgg16/cifar10/{experiment}/evaluations/curve.npz`
