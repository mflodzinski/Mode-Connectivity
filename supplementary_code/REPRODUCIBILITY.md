# Anonymous Reproducibility Supplement

This archive contains the implementation, frozen experiment configurations,
and execution scripts used for the paper. It intentionally excludes generated
checkpoints, downloaded benchmark data, logs, and result directories.

## Environment

The supported Python versions are 3.10 and 3.11. Install the locked
environment from the archive root:

    poetry install
    PYTHONPATH=.:src poetry run pytest

The reusable implementation is under src/mode_connectivity/. Experiment entry
points are under experiments/, frozen configurations are under
configs/experiments/, and local or Slurm launchers are under ops/.

## Data

The XOR inputs and labels are generated directly by the experiment code.
CIFAR-10 and Fashion-MNIST are obtained through torchvision; their preparation
tasks download the public datasets and freeze the train, validation, and
evaluation indices before training. No private data are used.

## Main experiment families

The controlled XOR experiments can be launched from the archive root with:

    poetry run python -m experiments.xor.train_linear_barriers
    poetry run python -m experiments.xor.permutation_scale
    poetry run python -m experiments.xor.curve_fitting
    poetry run python -m experiments.xor.joint_scale_polygonal

Their default argument lists and hyperparameter grids are stored under
configs/experiments/xor/. The corresponding cluster wrappers are under
ops/slurm/xor/.

The one-interior-point endpoint-scaling comparison reported in the paper uses
both endpoint scales, the absolute control-point parameterization, and a
501-point final evaluation grid. After generating the endpoint checkpoints,
submit the eight width/path-family jobs with:

```bash
for hidden_size in 2 3 5 7; do
  for curve_type in polygonal bezier; do
    sbatch --export=ALL,HIDDEN_SIZE="${hidden_size}",CURVE_TYPE="${curve_type}",SCALE_ENDPOINTS=both,INTERNAL_PARAMETERIZATION=absolute,INTERNAL_POINTS=1,EXPERIMENT_TAG=both_endpoints_k1,EVAL_POINTS=501 \
      ops/slurm/xor/run_joint_scale_polygonal.sh
  done
done
```

The wrapper fixes the remaining reported settings: no permutation, 1500 Adam
steps, learning rate 0.05, scale penalty `1e-4`, 31-point periodic fitting
grid, and one restart.

The larger-network endpoint training, alignment search, evaluation, and
reporting pipeline is configured under configs/experiments/training_stage/ and
configs/experiments/dense_linear_stage/. The complete resumable launch sequence
is documented in:

- ops/slurm/training_stage/README.md
- ops/slurm/dense_linear_stage/README.md
- ops/slurm/fashion_mnist/README.md

The default dense-linear configurations used for the reported
architecture--dataset combinations are:

- configs/experiments/dense_linear_stage/final_vgg11.yaml
- configs/experiments/dense_linear_stage/final_vgg13.yaml
- configs/experiments/dense_linear_stage/final_vgg16.yaml
- configs/experiments/dense_linear_stage/final_vgg19.yaml
- configs/experiments/dense_linear_stage/final_fashion_mnist.yaml

Scheduler resource flags are operational settings and may be changed without
altering the scientific configurations. The larger-network experiments are
computationally intensive and regenerate all endpoint checkpoints from
scratch.

## Third-party code

Vendored dependencies are under external/. Their upstream repositories, pinned
revisions, licenses, and local modifications are documented in
THIRD_PARTY_LICENSES.md and the corresponding
MODE_CONNECTIVITY_VENDOR_CHANGES.md files.
