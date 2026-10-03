# Slurm launchers

- `xor/`: controlled XOR sweeps.
- `training_stage/`: six independent CIFAR-10 VGG endpoint runs.
- `fashion_mnist/`: six independent Fashion-MNIST MLP endpoint runs.
- `dense_linear_stage/`: final-endpoint alignment grids, evaluation, status checks, and full-training-split reports. No cross-training-stage launcher remains active.
- `common.sh`: environment, path, and scheduler helpers.

Every submission family supports a dry-run path. Scheduler flags may be changed without changing the scientific configuration. Use the staged commands in [../../REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).
