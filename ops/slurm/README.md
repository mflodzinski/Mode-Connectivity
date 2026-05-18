# Slurm Launcher Surface

This directory contains the active Slurm-facing launcher wrappers for the retained experiment and verification workflows. Most scripts here are thin orchestration layers around `python -m experiments...` modules or small verification utilities.

## Shared Helper

- `common.sh`
  Centralizes repo-root resolution, virtualenv activation, `PYTHONPATH` setup, cache-directory setup, and a few shared helpers for invoking Python modules and upstream curve-evaluation scripts.

## Launcher Families

- `curves/`
  Train or evaluate Garipov-style curves and geometry variants.
- `endpoints/`
  Train or transform endpoint checkpoints used by downstream curve workflows.
- `lmc/`
  Run linear mode-connectivity training, packaging, benchmarking, and evaluation jobs.
- `sinkhorn/`
  Launch Sinkhorn-based alignment sweeps and comparison runs.
- `verification/`
  Run diagnostic or correctness checks on produced checkpoints or transforms.
- `xor/`
  Launch the retained XOR experiment flows on cluster infrastructure.
- `smoke/`
  Small end-to-end validation jobs for the refactored active surface.

## Environment Assumptions

These scripts generally assume:

- a usable Python environment, often via `VENV_ACTIVATE` or the default virtualenv path
- `PYTHONPATH` rooted at `src/` plus the repo root
- writable Matplotlib and cache directories, configured through `MPLCONFIGDIR` and `XDG_CACHE_HOME`
- required upstream submodules or checkpoint artifacts under `external/` when the corresponding workflow depends on them

## Relationship To Python Entry Points

- Slurm scripts are the cluster execution layer.
- The underlying experiment logic typically lives in `python -m experiments...`.
- Reusable implementation details live under `src/mode_connectivity`.
- Some curve evaluation flows still call vendored upstream scripts directly when the retained upstream tool is the source of truth.

## Related Guides

- [../../README.md](../../README.md)
- [../../experiments/README.md](../../experiments/README.md)
- [../README.md](../README.md)
- [smoke/README.md](smoke/README.md)
