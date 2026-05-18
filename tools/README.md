# Utility Tools

This directory contains operator-facing utilities that are useful after or around experiment runs. These scripts are not the main experiment entrypoints and are not intended to be the reusable library layer.

## Layout

- `plotting/`
  Scripts that turn checkpoints or result directories into comparison figures and summary plots.
- `verification/`
  Scripts that check properties of checkpoints, permutations, or transformed models.

## When To Use Tools Instead Of Experiment Runners

Use `tools/` when you already have produced artifacts and want to:

- generate plots from result directories or saved metrics
- inspect checkpoints or transformation outputs
- run focused diagnostics without launching a full experiment workflow

Use `experiments/` when you want to execute the primary training, evaluation, or sweep workflows themselves.

## Expected Inputs

Depending on the script, common inputs include:

- checkpoint files
- result directories under `results/`
- generated metric summaries or comparison artifacts

## Related Guides

- [../README.md](../README.md)
- [../experiments/README.md](../experiments/README.md)
- [../src/mode_connectivity/README.md](../src/mode_connectivity/README.md)
