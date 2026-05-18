# `mode_connectivity` Library

This package is the active reusable library layer for the repository. It holds experiment-independent logic for loading checkpoints, constructing models and loaders, evaluating paths, aligning networks, and wrapping the retained upstream dependencies.

## Package Map

- `common/`
  Shared path helpers, seed utilities, and config composition compatibility.
- `core/`
  Core runtime building blocks: checkpoint I/O, data loading, model creation, output helpers, and training command assembly.
- `evaluation/`
  Metric, interpolation, and evaluation routines used by multiple experiment families.
- `curves/`
  Shared helpers for Garipov-style curve training and curve analysis.
- `alignment/`
  Permutation specs, weight matching, Sinkhorn helpers, and path-alignment pipelines.
- `sinkhorn/`
  Shared VGG/CIFAR alignment logic and sweep support for Sinkhorn-based workflows.
- `lmc/`
  Reusable helpers for linear mode-connectivity training flows.
- `transform/`
  Function-preserving network transformations such as permutations and mirrors.
- `xor/`
  Reusable implementations behind the repo-level XOR runners.
- `analysis/`
  Small analysis helpers, especially plotting and alignment-analysis support.
- `external/`
  Adapter boundary for vendored upstream repositories under `external/`.
- `utils/`
  Small shared support modules, currently mostly CLI argument helpers.
- `metrics/`
  Convenience re-exports for common distance-reporting helpers.

## Import Boundaries

- Reusable logic belongs here in `src/mode_connectivity/`.
- Repo-facing runnable entrypoints belong in [../../experiments/README.md](../../experiments/README.md).
- Operator-facing one-off utilities belong in [../../tools/README.md](../../tools/README.md).
- `external/` is not the upstream code itself; it is the adapter layer that centralizes imports and path setup for vendored dependencies.

## Related Guides

- [../../README.md](../../README.md)
- [../../experiments/README.md](../../experiments/README.md)
- [../../configs/experiments/README.md](../../configs/experiments/README.md)
