# Local XOR Smoke Scripts

Disposable CPU-only smoke checks for the XOR experiments.

These are intended for local validation because the XOR workflows do not need
GPU or Slurm resources.

## Included scripts

- `xor_train_linear_minimal.sh`
  Trains a small retained set of XOR endpoints and writes pairwise linear-path
  diagnostics under `results/smoke_local/xor/train_linear`.
- `xor_permutation_scale_minimal.sh`
  Runs the reduced permutation-vs-scale comparison on the checkpoints produced
  by the training smoke script.
- `xor_joint_scale_path_minimal.sh`
  Runs a reduced path-only versus joint scale-and-Bezier comparison, including
  the artificially scaled positive control.
- `xor_joint_scale_polygonal_minimal.sh`
  Runs the corresponding reduced comparison for polygonal paths with multiple
  numbers of trainable internal bends.
- `run_xor_smoke_suite.sh`
  Runs all four scripts sequentially.

## Usage

From the repo root:

```bash
bash ops/local/smoke/run_xor_smoke_suite.sh
```

Or run them separately:

```bash
bash ops/local/smoke/xor_train_linear_minimal.sh
bash ops/local/smoke/xor_permutation_scale_minimal.sh
bash ops/local/smoke/xor_joint_scale_path_minimal.sh
bash ops/local/smoke/xor_joint_scale_polygonal_minimal.sh
```
