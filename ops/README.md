# Operations Scripts

This directory contains operator-facing shell entrypoints for running and validating the active experiment surface outside the reusable Python library.

## Layout

- `local/`
  Local scripts, primarily lightweight smoke checks that can run without cluster infrastructure.
- `slurm/`
  Cluster launcher wrappers and helper scripts for the active experiment families.

## When To Use Which

- Use `ops/local/` when you want a quick local validation pass, especially for small CPU-friendly workflows such as the XOR smoke checks.
- Use `ops/slurm/` when you want the maintained cluster execution surface for the active experiments and verification jobs.

## Existing Smoke Documentation

The smoke-specific directories already have their own focused guides:

- [local/smoke/README.md](local/smoke/README.md)
- [slurm/smoke/README.md](slurm/smoke/README.md)

## Related Guides

- [../README.md](../README.md)
- [slurm/README.md](slurm/README.md)
- [../experiments/README.md](../experiments/README.md)
