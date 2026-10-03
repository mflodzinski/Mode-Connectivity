# Fashion-MNIST deep-MLP connectivity benchmark

This family implements the two requested experiments with one shared set of
endpoint runs:

- **Experiment 4:** same-stage raw linear, permutation-aligned linear,
  permutation-plus-positive-scaling linear, and raw-endpoint nonlinear-path
  barriers at 16 checkpoints from initialization through epoch 100.
- **Experiment 5:** the three linear conditions at epoch 100, including a
  per-pair check of `raw > permutation > permutation+scale`.

The network is exactly `784 -> 512 x 10 -> 10`, with ReLU after each of the ten
hidden affine layers. It has no BatchNorm, dropout, residual connection, or
other architectural component that changes the clean permutation and positive
scaling symmetry group.

The checkpoint schedule is `0, 1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 40, 55, 70,
85, 100`. It matches the VGG training-stage experiment's count of 16 while
sampling the rapid early-training regime more densely and later training more
sparsely. All four connectivity conditions are evaluated at every checkpoint.

## Literature audit

The public [REPAIR paper](https://openreview.net/forum?id=gU5sJ6ZggcX) reports a
"10-layer MLP" Fashion-MNIST width experiment. Its public
[`Permute-Datasets-MLP.ipynb`](https://github.com/KellerJordan/REPAIR/blob/main/notebooks/Permute-Datasets-MLP.ipynb)
sets `layers=10`, but the class constructs an initial hidden ReLU layer and then
ten more hidden linear/ReLU pairs. The released code therefore contains eleven
hidden ReLU layers for that label. It evaluates width 512 among other widths,
uses 100-epoch checkpoints, normalizes Fashion-MNIST by `72.9404/255` and
`90.0212/255`, and its data pipeline includes horizontal flips and translations.
The repository does not include the Fashion-MNIST checkpoint-training routine
used by that evaluation notebook.

The appendix of
[Linear Mode Connectivity between Multiple Models modulo Permutation Symmetries](https://openreview.net/pdf/9d4174448944f5ca997041ff6ce3c5fd0b70f3ae.pdf)
specifies three hidden ReLU layers of width 512, Adam with learning rate
`1e-3`, batch size 512, at most 100 epochs, and no learning-rate scheduler for
MNIST/Fashion-MNIST.

This experiment follows the user's unambiguous **ten-hidden-layer** definition,
uses the newer paper's optimizer/batch/epoch recipe, retains REPAIR's dataset
normalization, and deliberately uses no augmentation. It is a documented hybrid
protocol, not an exact reproduction of either prior experiment.

## Run on Slurm

The full benchmark is compute intensive and is split into cluster-only
submissions that share one endpoint-training run:

```bash
bash ops/slurm/fashion_mnist/submit_endpoints.sh
bash ops/slurm/fashion_mnist/submit_linear_stage.sh
bash ops/slurm/fashion_mnist/submit_nonlinear_stage.sh
bash ops/slurm/fashion_mnist/submit_final_alignment.sh
```

See [`ops/slurm/fashion_mnist/README.md`](../../ops/slurm/fashion_mnist/README.md)
for the submitted dependency graph and resource overrides.

Use a new `output_root` after changing any scientific setting. Preparation
freezes the configuration, source hash, and stratified data splits. Test data
are inaccessible during endpoint training, alignment, and nonlinear-path
selection; they are opened only by final evaluation.

## Outputs

Each pair/stage directory contains the hard permutation, optimized exact
positive scales, selected quadratic Bezier path, and interpolation profiles.
`report/linear_stage/` contains the raw/WM/WM+scale time series and the final
hierarchy check. `report/nonlinear_stage/` contains only the raw-endpoint
quadratic-Bezier time series. `final_alignment/report/` compares raw, WM,
Sinkhorn, and Sinkhorn+scale on epoch-100 endpoints.
