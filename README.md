# Mode Connectivity Thesis Repository

This repository has been refactored to preserve the thesis experiments and results while separating reusable library code from runnable thesis workflows. The code backbone lives under `src/mode_connectivity/`, thesis-facing entrypoints live under `experiments/`, developer-facing plotting and verification CLIs live under `tools/`, and active Slurm launchers live in `ops/slurm/`.

## Active Layout

```text
.
├── thesis/
│   ├── tex/                  # Thesis source
│   ├── figures/aliases/      # Stable figure filenames used by draft.tex
│   ├── figures/source_index/ # Figure provenance index
│   └── MANIFEST.md           # Thesis experiment and figure traceability
├── src/mode_connectivity/    # Reusable Python library code only
├── experiments/             # Thesis experiment runners
├── tools/                   # Plotting and verification CLIs
├── ops/slurm/                # Active Slurm launchers grouped by domain
├── configs/                  # Active experiment configs
├── results/                  # Active result trees preserved in-place
├── plots/                    # Generated summary plots preserved in-place
├── external/                 # Vendored/submodule dependencies
├── tests/                    # Regression and smoke tests
└── archive/legacy/           # Historical, runtime, and low-priority preserved artifacts
```

## Thesis Source

- Root [`draft.tex`](./draft.tex) is now a compatibility wrapper.
- The editable thesis source lives at [`thesis/tex/draft.tex`](./thesis/tex/draft.tex).
- Figure filenames referenced by the thesis are materialized in [`thesis/figures/aliases`](./thesis/figures/aliases).
- Figure provenance is recorded in [`thesis/figures/source_index/figure_sources.json`](./thesis/figures/source_index/figure_sources.json).
- Experiment entrypoints are invoked as `python -m experiments...`.
- Plotting and verification utilities are invoked as `python -m tools...`.

## Thesis-Critical Experiment Families

- Curved-path VGG16 experiments:
  `experiments.curves.garipov_endpoints`,
  `experiments.curves.garipov_curve`
- Hyperplane-constrained geometry experiments:
  `experiments.curves.garipov_polygon`,
  `experiments.curves.random_plane`,
  `experiments.curves.symmetry_plane`
- Controlled LMC benchmark:
  `experiments.lmc.pytorch_vgg16_lmc_connected_pair`,
  `experiments.lmc.materialize_pytorch_vgg16_independent_pair`,
  `experiments.lmc.evaluate_pytorch_vgg_pair`,
  `experiments.lmc.evaluate_pytorch_vgg_split_suite`,
  `experiments.lmc.benchmark_alignment`
- XOR alignment experiments:
  `experiments.xor.permutation_scale`
- XOR connectivity experiments:
  `experiments.xor.basin_test`,
  `experiments.xor.train_linear_barriers`,
  `experiments.xor.curve_fitting`
- VGG Sinkhorn + scale chapter:
  `experiments.sinkhorn.vgg_cifar_alignment_sweep`,
  `experiments.sinkhorn.vgg_cifar_three_way_comparison`,
  `experiments.sinkhorn.vgg_cifar_three_way_barriers`
- Verification utilities:
  `tools.verification.network_transform`,
  `tools.verification.verify_shuffle_effect`,
  `tools.verification.check_model_functional_equivalence`

## Archive Policy

- Nothing thesis-relevant is deleted.
- If provenance or future usefulness is uncertain, the asset stays in place or is moved under `archive/legacy/`.
- Heavy result trees used by thesis experiments remain at their original paths unless compatibility is proven.
- `external/` submodules are intentionally left structurally unchanged.

## Archived Material

Archived material now lives under `archive/legacy/`, including:

- old weekly thesis update drafts and figures
- runtime outputs from `outputs/`, `wandb/`, and `.mplcache/`
- root-level loose artifacts and LaTeX build byproducts
- previously archived result trees such as `results/vgg16/cifar10/_archive`
- the pre-refactor `scripts/{analysis,train,plot,eval,experiments,lib,utils}` layout
- the pre-refactor `scripts/slurm` tree

## Known Gaps Preserved Explicitly

- `references.bib` is still missing from the repository.
- Some thesis figure names in `draft.tex` had no exact committed source image. Those aliases are preserved as explicit placeholders and are marked in the source index and `MANIFEST.md`.

## Config Layout

Active experiment configuration now lives only under `configs/experiments/`.
The tree is organized by family and composed through Hydra groups:

- `configs/experiments/curves/{_base,pairs,geometry,runs}`
- `configs/experiments/lmc/{_base,splits,runs,analysis}`
- `configs/experiments/sinkhorn/{_base,presets,runs}`
- `configs/experiments/xor/{runners,search}`

The legacy `configs/garipov`, `configs/pytorch_vgg`, and `configs/analysis`
trees are no longer part of the active runtime surface.

## Slurm Layout

Active Slurm launchers are now small generic wrappers grouped by domain:

- `ops/slurm/curves/`
- `ops/slurm/endpoints/`
- `ops/slurm/lmc/`
- `ops/slurm/sinkhorn/`
- `ops/slurm/xor/`
- `ops/slurm/verification/`

The old per-run submit script sprawl was removed. The remaining scripts are
parameterized entrypoints that accept config names, checkpoint paths, and a
small set of environment overrides.
