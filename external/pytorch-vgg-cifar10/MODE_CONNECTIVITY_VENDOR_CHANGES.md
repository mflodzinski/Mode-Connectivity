# Vendor Changes For Mode-Connectivity

This vendored directory was imported from an external source tree. At the time this note was added, that source tree was clean and pinned at commit `03800cf` on `master`.

## Comparison Basis

This vendored directory did not have a separate `upstream` remote configured locally before it was imported. In the visible local history, the original project lineage ends at commit `9c5da95`, and all later commits are local changes for this repository. Those local changes are intended to be kept as a single squashed layer on top of that base.

## What Changed Locally

- `main.py`
  The training script was adapted from a simpler single-run baseline into a more reusable endpoint generator for the parent repo.
  Visible local behavior in the current tree includes:
  - deterministic seeding
  - explicit `--seed`, `--save-every`, and `--epoch-print-freq` controls
  - periodic checkpoint emission (`checkpoint_<epoch>.tar`)
  - separate best and final checkpoint outputs
  - checkpoint loading and saving logic aligned with downstream consumption
- `run.sh`
  The launcher was turned into a small multi-seed training helper.
  In the current pinned state it:
  - selects an architecture through `ARCH`
  - trains the same model for seeds `0` and `1`
  - writes per-seed output directories like `save_vgg11_seed0`
- `.gitignore`
  The vendored directory ignores extracted state-dict checkpoints that are generated for downstream reuse in the parent repo.

## Main Places To Inspect

- `main.py`
  Main training-loop and checkpoint-format differences.
- `run.sh`
  Multi-seed orchestration and architecture selection.
- `.gitignore`
  Ignore rule for extracted checkpoint artifacts.

## Scope Of The Squashed Local Layer

The single local-layer commit on top of `9c5da95` is expected to cover the retained changes summarized above, rather than preserve the intermediate local-only commit history.
