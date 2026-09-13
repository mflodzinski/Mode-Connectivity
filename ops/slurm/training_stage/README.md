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
