# Smoke Slurm Pack

Disposable Slurm launchers for validating that the refactored experiment
surface still runs end to end after cleanup.

All jobs write under `results/smoke/` and are intended to be deleted once the
refactor is trusted on the cluster.

These launchers are intentionally resource-thin smoke checks:
- `--cpus-per-task=1` everywhere
- reduced wall times and memory caps
- GPUs are still requested only where the retained runner path expects them

## Included checks

- `curve_minimal.sh`
  Runs a Bezier curve training job for 2 epochs between existing VGG16
  endpoints and evaluates the resulting checkpoint.
- `sinkhorn_minimal.sh`
  Runs a single Sinkhorn alignment sweep combination for 50 alignment
  iterations between existing VGG11/CIFAR checkpoints.
- `lmc_resume_minimal.sh`
  Runs the pytorch-vgg split-training workflow from scratch with
  `shared_epochs=0` and `final_epochs=1` so the smoke check is self-contained.
- `lmc_benchmark_minimal.sh`
  Runs the alignment benchmark against the checkpoints produced by
  `lmc_resume_minimal.sh`.
- `xor_train_linear_minimal.sh`
  Trains a small batch of XOR models and writes linear interpolation results
  for the retained pairs.
- `xor_permutation_scale_minimal.sh`
  Runs the XOR permutation-vs-scale experiment on the checkpoints produced by
  `xor_train_linear_minimal.sh`.
- `submit_smoke_suite.sh`
  Convenience submitter that chains the dependent jobs.

## Suggested usage

Submit individually:

```bash
sbatch ops/slurm/smoke/curve_minimal.sh
sbatch ops/slurm/smoke/sinkhorn_minimal.sh
sbatch ops/slurm/smoke/lmc_resume_minimal.sh
sbatch ops/slurm/smoke/xor_train_linear_minimal.sh
```

Or submit the chained suite:

```bash
bash ops/slurm/smoke/submit_smoke_suite.sh
```

## Expected outputs

- `results/smoke/curves/...`
- `results/smoke/sinkhorn/...`
- `results/smoke/lmc/...`
- `results/smoke/xor/...`
