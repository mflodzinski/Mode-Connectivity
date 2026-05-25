# Mode-Connectivity

This repository contains the active code, configs, and operator scripts for the mode-connectivity thesis workflows: Garipov-style curves, linear mode connectivity, Sinkhorn-based alignment, and smaller XOR studies. The reusable implementation lives under `src/mode_connectivity`, while repo-facing runners, Slurm launchers, and plotting or verification utilities live alongside it in dedicated top-level directories.

## Setup

- Python: `>=3.10,<3.12`
- Dependency management: Poetry
- External code: vendored third-party directories under `external/`

Typical setup:

```bash
poetry install
poetry run pytest
```

Notes:

- `poetry.toml` is configured for an in-project virtual environment, so Poetry will create `.venv/`.
- `data/`, `results/`, and `plots/` are working directories for datasets and generated artifacts, not the primary source tree.
- `external/` contains vendored upstream code. Treat it as dependency code, not the main place to edit repo logic. Third-party license details are listed in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Where To Start

- Run an experiment:
  See [experiments/README.md](experiments/README.md) and [configs/experiments/README.md](configs/experiments/README.md).
- Inspect reusable library code:
  Start at [src/mode_connectivity/README.md](src/mode_connectivity/README.md).
- Use local or cluster operator scripts:
  See [ops/README.md](ops/README.md) and [ops/slurm/README.md](ops/slurm/README.md).
- Use plotting or verification utilities:
  See [tools/README.md](tools/README.md).

Common active entry surfaces:

- `python -m experiments...`
- `python tools/...`
- `bash ops/local/...`
- `sbatch` or `bash` under `ops/slurm/...`

## Top-Level Layout

- `src/`
  Active reusable Python library code.
- `experiments/`
  Repo-facing runnable entrypoints for experiment families.
- `configs/`
  Canonical configuration trees, primarily under `configs/experiments/`.
- `ops/`
  Local smoke scripts and Slurm launcher wrappers.
- `tools/`
  Operator-facing plotting and verification utilities.
- `tests/`
  Active structure and behavior smoke tests for the retained layout.
- `external/`
  Vendored upstream repositories used by active workflows.

## Directory Guides

- [src/mode_connectivity/README.md](src/mode_connectivity/README.md)
- [experiments/README.md](experiments/README.md)
- [configs/experiments/README.md](configs/experiments/README.md)
- [ops/README.md](ops/README.md)
- [ops/slurm/README.md](ops/slurm/README.md)
- [tools/README.md](tools/README.md)
