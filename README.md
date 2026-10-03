# Scaling Symmetries and Mode Connectivity

This is the reproducibility repository for **“The Role of Scaling Symmetries in Linear and Nonlinear Mode Connectivity of Neural Networks.”** The paper source of truth is [`weekly_thesis_update(4)/paper_aistats2027/main.tex`](<weekly_thesis_update(4)/paper_aistats2027/main.tex>), with the compiled manuscript at [`main.pdf`](<weekly_thesis_update(4)/paper_aistats2027/main.pdf>).

The repository reproduces four reported experiment groups:

1. exhaustive permutation tests for width-2 XOR networks, including the minibatch-SGD robustness check;
2. exhaustive permutation and positive-scale alignment for XOR widths 3, 5, and 7;
3. nonlinear polygonal and quadratic Bézier paths, with and without endpoint scaling;
4. final-endpoint alignment for VGG11/13/16/19 on CIFAR-10 and a 10-layer, width-512 MLP on Fashion-MNIST.

The exact paper-to-code/result mapping and end-to-end commands are in [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Installation

Python 3.10 or 3.11 and Poetry are supported.

```bash
poetry install
PYTHONPATH=.:src poetry run pytest -q
```

The lockfile is committed. XOR data are generated in code. CIFAR-10 and Fashion-MNIST are downloaded through `torchvision`; no private data or pretrained checkpoints are required. Vendored third-party code and pinned revisions are documented in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

## Quick checks

Run the CPU-only XOR smoke suite:

```bash
bash ops/local/smoke/run_xor_smoke_suite.sh
```

Inspect cluster commands without submitting jobs:

```bash
bash ops/slurm/training_stage/submit_endpoints.sh --dry-run \
  output_root=results/training_stage_vgg11_final model=VGG11

bash ops/slurm/dense_linear_stage/submit_all.sh --dry-run \
  --config-name dense_linear_stage/final_vgg11
```

## Layout

- `experiments/`: thin runnable entry points for the paper experiments.
- `configs/experiments/`: frozen scientific configurations and XOR search grids.
- `src/mode_connectivity/`: reusable training, alignment, evaluation, and reporting code.
- `ops/`: local smoke tests and resumable Slurm launchers.
- `tools/plotting/`: scripts that regenerate every empirical paper figure.
- `tests/`: unit, import, config-composition, and shell-syntax checks.
- `external/`: vendored upstream dependencies.
- `results/`: retained reported artifacts in this working copy (ignored for normal Git commits).
- `archive/`: historical code and non-paper results; nothing there is imported by active code.
- `weekly_thesis_update(4)/paper_aistats2027/`: manuscript, bibliography, styles, and paper figures.

The historical module name `dense_linear_stage` now contains only the paper's
final-endpoint alignment workflow; the cross-training-stage mode has been
removed from the active tree. Generated datasets, checkpoints, results, plots,
and new manuscript build artifacts are intentionally ignored by Git. Existing
tracked manuscript sources remain tracked. Use a new `output_root` whenever a
scientific configuration changes; the cluster pipelines freeze a protocol hash
and reject incompatible reuse.
