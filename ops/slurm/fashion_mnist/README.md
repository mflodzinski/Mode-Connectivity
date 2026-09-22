# Fashion-MNIST deep-MLP Slurm jobs

Run these commands on the cluster login node from the repository checkout.
The stages are deliberately separate so the six endpoint runs are shared.

First submit dataset preparation and endpoint training:

```bash
bash ops/slurm/fashion_mnist/submit_endpoints.sh --dry-run
bash ops/slurm/fashion_mnist/submit_endpoints.sh
```

After every endpoint task has completed, submit linear stage connectivity. This
runs raw, weight-matched, and weight-matched-plus-scale interpolation at all 16
checkpoints:

```bash
bash ops/slurm/fashion_mnist/submit_linear_stage.sh --dry-run
bash ops/slurm/fashion_mnist/submit_linear_stage.sh
```

Submit nonlinear stage connectivity independently. These jobs fit and evaluate
only raw-endpoint quadratic Bezier paths; they do not run permutation or scale
alignment:

```bash
bash ops/slurm/fashion_mnist/submit_nonlinear_stage.sh --dry-run
bash ops/slurm/fashion_mnist/submit_nonlinear_stage.sh
```

Finally, after the epoch-100 endpoints exist, submit the separate final-endpoint
comparison of WM, Sinkhorn, and Sinkhorn followed by positive-scale refinement:

```bash
bash ops/slurm/fashion_mnist/submit_final_alignment.sh --dry-run
bash ops/slurm/fashion_mnist/submit_final_alignment.sh
```

The launchers create these jobs:

1. `fmnist_prepare`: CPU job that downloads Fashion-MNIST and freezes the
   protocol and data splits.
2. `fmnist_train`: six-task GPU array, one independently initialized endpoint
   per task; each run saves the same number of checkpoints as the VGG protocol.
3. `fmnist_linear_stage`: 48-task GPU array plus a CPU report job.
4. `fmnist_nonlinear_stage`: a separate 48-task GPU array plus a CPU report.
5. `fmnist_final_sinkhorn` and `fmnist_final_scale`: two three-task GPU arrays,
   followed by a CPU final-alignment report.

The default requests one A40 GPU, 8 GB of RAM, and at most four simultaneous
array tasks. Cluster-specific resources can be supplied without editing files:

```bash
bash ops/slurm/fashion_mnist/submit_endpoints.sh \
  --partition=gpu --qos=short --gres=gpu:a40:1 --account=MY_ACCOUNT \
  --concurrency=4
```

Hydra overrides follow the launcher arguments. Use a new `output_root` whenever
the scientific protocol changes:

```bash
bash ops/slurm/fashion_mnist/submit_endpoints.sh \
  output_root=results/fashion_mnist_mlp10x512_run2
```

Pass the identical `output_root` override to every later launcher for that run.

The checkout must provide either `VENV_ACTIVATE` or the default virtual
environment at `$HOME/venvs/mode-connectivity/bin/activate`. All training and
analysis happens inside Slurm allocations; each launcher itself only validates
artifacts, writes a manifest, and calls `sbatch`. Analysis tasks resume at
durable alignment or nonlinear-path pass boundaries after preemption.
