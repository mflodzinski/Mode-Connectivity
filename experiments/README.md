# Paper experiment entry points

These modules are the runnable surface for the experiments reported in the AISTATS paper. Reusable implementation lives in `src/mode_connectivity/`.

- `xor/`: exhaustive permutation, permutation/scale, and nonlinear-path studies.
- `training_stage/`: independent CIFAR-10 VGG endpoint training used by the final benchmark.
- `fashion_mnist/`: independent Fashion-MNIST MLP endpoint training.
- `dense_linear_stage/`: final-endpoint validation search, frozen evaluation, status reporting, and full-training-split evaluation for the five larger-network settings. The directory name is retained for import compatibility; it no longer exposes cross-stage experiments.

Run modules as `python -m experiments.<family>.<entrypoint>`. See [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for the exact command and config associated with each figure and table.
