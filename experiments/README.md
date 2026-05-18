# Experiment Runners

This directory contains the repo-facing runnable entrypoints for the active experiment families. These modules are the intended `python -m experiments...` surface and delegate most reusable logic to `src/mode_connectivity`.

## Families

- `curves/`
  Garipov-style curve training, endpoint preparation, and constrained geometry variants.
- `lmc/`
  Linear mode-connectivity training, packaging, benchmarking, and evaluation flows.
- `sinkhorn/`
  VGG/CIFAR alignment sweeps and comparison runs built around Sinkhorn-based rebasining.
- `xor/`
  Thin wrappers around retained argparse-heavy XOR experiments.

## How Runners Pair With Configs

- Config-driven families:
  `curves/`, `lmc/`, and `sinkhorn/` compose defaults from `configs/experiments/...` and then execute through reusable library helpers.
- Thin CLI wrapper family:
  `xor/` loads preset argv-style settings from `configs/experiments/xor/runners/` and forwards them to the retained XOR implementations.

See [../configs/experiments/README.md](../configs/experiments/README.md) for the config tree layout.

## Outputs

At a high level, these runners write generated artifacts under `results/`, typically grouped by experiment family:

- curve and geometry outputs under `results/.../curves/...`
- LMC outputs under `results/.../lmc/...`
- Sinkhorn outputs under `results/.../sinkhorn/...`
- XOR outputs under `results/xor/...`

## Related Guides

- [../README.md](../README.md)
- [../src/mode_connectivity/README.md](../src/mode_connectivity/README.md)
- [../ops/slurm/README.md](../ops/slurm/README.md)
