# Training-stage connectivity jobs

Run these commands from the repository root on a DAIC login node. They compose
the frozen protocol in `configs/experiments/training_stage/default.yaml`.

```bash
# Inspect the complete pilot submission without creating results or jobs.
bash ops/slurm/training_stage/submit_pilot.sh --dry-run

# Queue preparation, the six independent endpoint runs, three pilot pairs,
# their audit, and the gate that verifies the 4 GB memory target.
bash ops/slurm/training_stage/submit_pilot.sh

# After the pilot succeeds, queue unfinished work. Completed and live jobs are
# reused from results/training_stage/submissions.json.
bash ops/slurm/training_stage/submit_main.sh
```

`submit_all.sh` queues the same dependency graph in one invocation. Training
jobs do not wait for the pilot gate; bulk alignment does. Every generated job
uses two CPUs and 4 GB host RAM. GPU jobs request one A40, and every wall time is
strictly below four hours.

Useful submission flags are `--concurrency`, `--partition`, `--qos`, `--gres`,
and `--account`. Hydra overrides follow those flags, for example:

```bash
bash ops/slurm/training_stage/submit_pilot.sh --dry-run \
  output_root=/absolute/scratch/path/training_stage \
  data_root=/absolute/shared/path/cifar10
```

The first real submission downloads CIFAR-10 once in the preparation job and
freezes the split indices, configuration, source hash, and dependency manifest.
Changing a scientific setting or experiment source after that point requires a
new `output_root`. A task writes a complete status only after all expected
artifacts exist. Re-running a submission script queues only missing or failed
tasks and resumes their recovery checkpoints.

Inspect `results/training_stage/status/`, `logs/`, and `submissions.json` while
the experiment runs. The final report contains aggregate JSON/CSV data, heatmap
figures, absolute interpolation profiles, audit discrepancies, and available
Slurm accounting output.

To train endpoints without scheduling any alignment work, use
`submit_endpoints.sh`. This is used by the nonlinear training-stage experiment
when additional seed pairs are missing.

## Alignment-only VGG11 parameter retry

If the VGG11/CIFAR-10 endpoints are already complete, rerun only Sinkhorn and
fixed-permutation Sinkhorn+scale with a validation-only grid search:

```bash
bash ops/slurm/training_stage/submit_alignment_grid.sh \
  results/training_stage_vgg11_pair01_atol2e5 \
  results/training_stage_vgg11_alignment_grid
```

The default Sinkhorn grid is learning rate `{0.005, 0.01, 0.05}` by temperature
`{1.0, 1.5}` with `l=1`. After selecting one global setting across all stage
pairs, the scale-only grid is learning rate `{0.01, 0.05}` by scale penalty
`{1e-4, 1e-3, 1e-2}`. Override these with the `BASE_LRS`, `TAUS`,
`SINKHORN_L`, `SCALE_LRS`, and `LAMBDA_SCALES` environment variables.

The job reuses the source `endpoints/` directory and frozen subset indices; it
does not submit training or inspect test data. Rankings are written to
`selected_base.json` and `selected_scale.json`. The chosen artifacts are linked
under `selected/pairs/`.

After both grids and selections complete, evaluate only the globally selected
artifacts on the frozen train/test evaluation subsets:

```bash
bash ops/slurm/training_stage/submit_alignment_grid_evaluation.sh \
  results/training_stage_vgg11_pair01_atol2e5 \
  results/training_stage_vgg11_alignment_grid
```

This submits 16 short GPU evaluations followed by a CPU report. It does not
rerun alignment. Matrices, barrier tables, improvements, and absolute profiles
are written under `selected/report/`.

## Git Re-Basin CIFAR-10 MLP reproduction and extension

The MLP preset trains only seed pair `(0, 1)`. It reproduces the Figure 3
same-stage weight-matching curve at all 100 paper checkpoints, adds a true
initialization measurement, and runs the 4-by-4 cross-stage experiment at
completed epochs 1, 10, 50, and 100.

```bash
# First inspect, then submit the engineering pilot.
bash ops/slurm/training_stage/submit_mlp_pilot.sh --dry-run
bash ops/slurm/training_stage/submit_mlp_pilot.sh

# After status/gate.json says complete, submit all unfinished main tasks.
bash ops/slurm/training_stage/submit_mlp_main.sh
```

The default output is `results/training_stage_git_rebasin_mlp_pair01`. Use a
new directory for every protocol change, for example:

```bash
bash ops/slurm/training_stage/submit_mlp_pilot.sh \
  output_root=results/training_stage_git_rebasin_mlp_pair01_rerun
```

The pilot contains 19 logical tasks: preparation, smoke check, two independent
training jobs, four onset checkpoints, three base/scale/continuation alignment
branches, the dense audit, and the gate. Slurm arrays may make the number of
rows shown by `squeue` differ from the number of submission commands. The full
DAG contains 174 logical tasks, of which 101 are independent onset evaluations
and 64 are the four cross-stage alignment/evaluation phases.
