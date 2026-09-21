# Nonlinear training-stage connectivity

This experiment fits raw-endpoint nonlinear paths between VGG11 checkpoints.
It never loads permutation, Sinkhorn, or scale artifacts from the source runs.
Run all commands from the repository root on a DAIC login node.

## 1. Ensure all endpoint checkpoints exist

The default nonlinear configuration reuses seeds 0 and 1 from
`results/training_stage_vgg11_pair01_atol2e5`. If seeds 2–5 have not been
trained, submit only the endpoint portion of the existing training-stage DAG:

```bash
bash ops/slurm/training_stage/submit_endpoints.sh \
  'seed_pairs=[[2,3],[4,5]]' \
  output_root=results/training_stage_vgg11_pairs2345
```

Wait for its four `stage_train` jobs to finish. Preparation and smoke are two
additional prerequisite tasks. The nonlinear preparation job refuses to start
if any required source checkpoint is missing or if the source subset indices
differ.

## 2. Engineering pilot

If the initial 5,000-example pilot fails its final-final positive control, run
the focused equal-budget calibration before changing the main protocol:

```bash
bash ops/slurm/nonlinear_stage/submit_calibration.sh --dry-run
bash ops/slurm/nonlinear_stage/submit_calibration.sh
```

This diagnostic submits six final-final Bezier fits on the 43,000 training examples that
are disjoint from `train_eval`. Each fit uses 25 passes (1.075M examples), with
learning rates `0.005`, `0.015`, and `0.03`, both with and without path weight
decay. All trials use the same random stream. Results are written to
`results/nonlinear_stage_vgg11_calibration/calibration/results.json` and
`barriers.png`. These 25-pass equal-budget trials diagnose the effect of data
coverage; they do not replace the established 200-pass protocol. Use a
separate output root for any subsequent calibration.

The primary protocol fits both Bezier and confirmation polygon paths for 200
passes on `curve_fit` using learning rate `0.015` and path weight decay
`5e-4`. The 43,000-example `curve_fit` set excludes the frozen 2,000-example
training evaluation set.

Inspect the commands, then submit the 28 logical pilot tasks:

```bash
bash ops/slurm/nonlinear_stage/submit_pilot.sh --dry-run
bash ops/slurm/nonlinear_stage/submit_pilot.sh
```

The pilot consists of preparation, smoke, 12 fits, 12 validation evaluations,
one dense audit, and one gate. Fits and evaluations are Slurm arrays, so
`squeue` may show fewer submission rows. Wait until the queue is empty and
check:

```bash
python - <<'PY'
import json
print(json.load(open("results/nonlinear_stage_vgg11/status/gate.json"))["status"])
PY
```

Do not continue unless this prints `complete`.

If an architecture-specific error-barrier expectation is explicitly rejected
after reviewing the validation-only pilot, record the decision without erasing
the failed gate or accessing test data:

```bash
bash ops/slurm/nonlinear_stage/accept_pilot.sh \
  results/nonlinear_stage_vgg11_fullfit \
  --family bezier --restart 1
```

This requires the selected candidate to pass the configured loss threshold,
reruns the dense-grid and resource checks, saves the original gate as
`status/gate.failed.json`, and records the waived error threshold and reason in
the replacement `gate.json`.

## 3. Primary paths

```bash
bash ops/slurm/nonlinear_stage/submit_main.sh --dry-run
bash ops/slurm/nonlinear_stage/submit_main.sh
```

This reuses the pilot artifacts and submits the remaining primary Bézier paths,
their validation-only evaluations, and the deterministic screening task. Wait
for `status/screen.json` to become complete. Test data has not been opened at
this point.

## 4. Confirmation and final evaluation

```bash
bash ops/slurm/nonlinear_stage/submit_confirm.sh --dry-run
bash ops/slurm/nonlinear_stage/submit_confirm.sh
```

This command reads the frozen `confirmation_targets.json`, submits additional
restarts only for flagged pairs, freezes one validation-selected path per pair,
then submits test evaluation and reporting. Re-running any submission command
is safe: completed and currently live tasks are reused, while incomplete tasks
resume from recovery state.

`submit_all.sh` is stage-aware. Before screening exists it queues work through
screening; run it a second time after screening to queue confirmation and final
evaluation.

Useful scheduler flags are `--concurrency`, `--partition`, `--qos`, `--gres`,
and `--account`. Hydra overrides follow these flags. Use the same overrides on
every phase because the first preparation freezes them into the protocol. Every
task requests two CPUs and 4 GB host RAM; GPU tasks request one A40.

For a one-pair exploratory run, use a separate output root and override the
source list consistently in every command:

```bash
bash ops/slurm/nonlinear_stage/submit_pilot.sh \
  output_root=results/nonlinear_stage_vgg11_pair01 \
  'source_pairs=[{seeds:[0,1],root:results/training_stage_vgg11_pair01_atol2e5}]'
```
