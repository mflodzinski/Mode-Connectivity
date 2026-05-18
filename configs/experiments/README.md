# Experiment Config Tree

This directory is the canonical configuration tree for active experiment runs. The main experiment families use Hydra-style composition, while the XOR wrappers use smaller OmegaConf-loaded preset files.

## Layout Conventions

- `_base/`
  Shared defaults and reusable config fragments for a family.
- `runs/`
  Concrete runnable experiment configs.
- `pairs/`
  Endpoint-pair definitions for curve experiments.
- `geometry/`
  Geometry or path-shape variants for curve experiments.
- `presets/`
  Reusable named parameter presets, currently used by Sinkhorn runs.
- `splits/`
  Reusable split definitions for LMC workflows.
- `runners/`
  XOR runner presets that define argv-like defaults for thin wrapper modules.
- `search/`
  XOR search-space or sweep-oriented preset files.

## Family Notes

- `curves/`
  Uses composition across `_base/`, `geometry/`, `pairs/`, and `runs/`.
- `lmc/`
  Uses `_base/`, `splits/`, `analysis/`, and `runs/`.
- `sinkhorn/`
  Uses `_base/`, `presets/`, and `runs/`.
- `xor/`
  Stores preset files consumed directly by the thin XOR wrappers rather than full Hydra family composition.

## Typical Composition Pattern

A representative curve run such as `curves/runs/curve_seed0_seed1_reg.yaml` composes:

- shared defaults from `curves/_base/common`
- curve-training settings from `curves/_base/curve_training`
- a geometry choice from `curves/geometry/...`
- an endpoint pair from `curves/pairs/...`
- final run-specific overrides in `runs/...`

## Where To Edit

- Shared defaults:
  edit the family `_base/` directory.
- Dataset or model pairings for curve-style runs:
  edit `curves/pairs/`.
- Concrete runnable experiment definitions:
  edit the relevant family `runs/` directory.
- Geometry-specific curve behavior:
  edit `curves/geometry/`.
- Sinkhorn reusable presets:
  edit `sinkhorn/presets/`.
- LMC split definitions:
  edit `lmc/splits/`.
- XOR CLI argument presets:
  edit `xor/runners/`.

## Related Guides

- [../../README.md](../../README.md)
- [../../experiments/README.md](../../experiments/README.md)
