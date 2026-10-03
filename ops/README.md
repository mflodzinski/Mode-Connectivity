# Operations Scripts

This directory contains operator-facing shell entrypoints for running and validating the active experiment surface outside the reusable Python library.

## Layout

- `local/`
  Lightweight XOR smoke checks that run without cluster infrastructure.
- `slurm/`
  Cluster launchers for the XOR and larger-network paper experiments.

## When To Use Which

- Use `ops/local/` when you want a quick local validation pass, especially for small CPU-friendly workflows such as the XOR smoke checks.
- Use `ops/slurm/` for the maintained cluster execution surface for endpoint training, final alignment, and XOR sweeps.

## Existing Smoke Documentation

The smoke-specific directories already have their own focused guides:

- [local/smoke/README.md](local/smoke/README.md)
- [../REPRODUCIBILITY.md](../REPRODUCIBILITY.md)

## Related Guides

- [../README.md](../README.md)
- [slurm/README.md](slurm/README.md)
- [../experiments/README.md](../experiments/README.md)
