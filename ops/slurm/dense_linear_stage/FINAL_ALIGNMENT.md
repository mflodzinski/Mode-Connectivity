# Final-endpoint alignment benchmark

This benchmark evaluates all six alignment conditions at the final checkpoint
for three disjoint independently trained endpoint pairs: `(0,1)`, `(2,3)`, and
`(4,5)`. Each optimized method receives a pair-specific validation-only grid
search. Test data are first loaded after the selected hyperparameters and
method artifacts have been frozen.

Six endpoints are intentional. Three endpoints would yield the three pairs
`(0,1)`, `(0,2)`, and `(1,2)`, but those observations share models and are not
independent replicates for a sample standard deviation.

## 1. Inventory endpoints on DAIC

From the repository root:

```bash
python -m experiments.dense_linear_stage.audit_endpoints
```

The `three_pairs` column must say `YES` for one internally consistent training
root per architecture. Do not combine checkpoints from different recipes just
because their seed numbers differ.

## 2. Train missing VGG endpoint sets

Preview first, then submit only the architectures that do not have seeds 0--5:

```bash
bash ops/slurm/training_stage/submit_endpoints.sh --dry-run model=VGG13 output_root=results/training_stage_vgg13_final
bash ops/slurm/training_stage/submit_endpoints.sh model=VGG13 output_root=results/training_stage_vgg13_final
```

Repeat with `VGG16` and `VGG19`. Re-running the same command is resume-safe.
The final checkpoints are written as
`results/training_stage_vgg*_final/endpoints/SEED/epoch_200.pt`.

## 3. Preview the alignment DAGs

Each optimized method has 12 candidates split over three parallel jobs:

- `WM+scale`: one scale-learning-rate per job (`0.005`, `0.02`, or `0.08`),
  each with penalties `{1e-4, 1e-3, 3e-3, 1e-2}`.
- `Sinkhorn`: one temperature per job (`1.0`, `2.5`, or `5.0`), each with
  learning rates `{0.03, 0.10, 0.30, 0.75}`.
- `joint Sinkhorn+scale`: one learning rate per job (`0.01`, `0.05`, or
  `0.20`), each with temperature `{1.0, 2.5}` crossed with scale penalty
  `{3e-4, 3e-3}`.
- `Sinkhorn->scale`: the same three scale-learning-rate jobs and four penalties
  as `WM+scale`, starting from that pair's selected Sinkhorn artifact.

Print the exact Slurm task, seed pair, candidate indices, and values before
submission:

```bash
python -m experiments.dense_linear_stage.grid_plan \
  --config-name dense_linear_stage/final_vgg11
```

Every candidate writes to its own `method/candidate` directory. Selection waits
for all three jobs, then performs one full-budget refit of the validation-best
configuration. Parallel jobs therefore never write the same candidate artifact.

```bash
for name in final_vgg11 final_vgg13 final_vgg16 final_vgg19 final_fashion_mnist; do
  bash ops/slurm/dense_linear_stage/submit_main.sh --dry-run --config-name "dense_linear_stage/${name}"
done
```

If a complete architecture already lives in a different compatible
training-stage root, override all six `source_roots` entries on the submission
command instead of copying checkpoints.

## 4. Submit

Submit one architecture first and inspect its resource usage before launching
the remainder:

```bash
bash ops/slurm/dense_linear_stage/submit_main.sh --config-name dense_linear_stage/final_vgg11
bash ops/slurm/dense_linear_stage/status.sh results/final_alignment_vgg11_cifar10
```

Then submit the remaining configurations:

```bash
for name in final_vgg13 final_vgg16 final_vgg19 final_fashion_mnist; do
  bash ops/slurm/dense_linear_stage/submit_main.sh --config-name "dense_linear_stage/${name}"
done
```

Submission is resume-safe: running the same command again schedules only
missing or failed tasks. Candidate fitting uses `val_tune` for early stopping,
candidate selection uses the disjoint `val_select` split, and the final report
uses the complete official test set.

## 5. Monitor and collect

```bash
squeue -u "$USER"
bash ops/slurm/dense_linear_stage/status.sh results/final_alignment_vgg13_cifar10
```

The final values are in each experiment's `report/aggregates.json` and
`report/barriers.csv`. `aggregates.json` contains the three pair values plus
their mean and sample standard deviation. The report also produces a paired
training/test loss-barrier plot so a zero test barrier cannot hide a positive
training-objective barrier. Copy the compact reports locally:

```bash
rsync -av --include='*/' --include='report/***' --exclude='*' \
  mlodzinski@login.daic.tudelft.nl:/tudelft.net/staff-bulk/ewi/insy/PRLab/Students/mlodzinski/Mode-Connectivity/results/final_alignment_*/ \
  results/
```
