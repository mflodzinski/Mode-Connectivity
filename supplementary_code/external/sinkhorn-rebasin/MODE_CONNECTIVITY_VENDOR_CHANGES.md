# Vendor Changes For Mode-Connectivity

This vendored directory was imported from an external source tree. At the time this note was added, that source tree was clean and pinned at commit `68e886f` on `main`.

## Comparison Basis

This vendored directory did not have a separate `upstream` remote configured locally before it was imported. In the visible local history, the original project lineage ends at commit `1c601d4`, and all later commits are local changes for this repository. Those local changes are intended to be kept as a single squashed layer on top of that base.

## What Changed Locally

- Scale-compensated rebasining support was added for the LMC CNN workflow.
  The retained implementation now supports learned scale factors alongside permutations and compensates adjacent weights and biases so functions stay unchanged under rescaling.
- `rebasin/rebasinnet/scale_utils.py`
  New helper module for:
  - converting log-scales to scale / inverse-scale vectors
  - applying output scaling to linear or convolutional weights
  - applying inverse input scaling to downstream weights
  - transforming biases consistently
- `rebasin/rebasinnet/symmnet.py`
  Main local integration point for scale-aware transforms inside the rebasining network.
  This is where scale and inverse-scale values are unpacked and applied during training-time and eval-time rebasining.
- `examples/main_lmc_cnn.py`
  Example workflow updated to exercise the scale-aware path.
- `tests/test_scale_compensation.py`
  Added regression tests that verify scale compensation preserves linear and convolutional functions.
- `rebasin/loss/loss.py`
  Local history also shows adjustments around the loss surface, including cosine-similarity-related additions and follow-up edits.
- `rebasin/rebasinnet/graph/graph.py`
  Earlier local history includes a compatibility fix around `torch.nn.flatten`.

## Main Places To Inspect

- `rebasin/rebasinnet/symmnet.py`
  Core scale-aware rebasining logic.
- `rebasin/rebasinnet/scale_utils.py`
  Helper functions introduced by these local changes.
- `tests/test_scale_compensation.py`
  Behavior-preservation coverage for the scale-compensation path.
- `examples/main_lmc_cnn.py`
  Example integration point for the new behavior.
- `rebasin/loss/loss.py`
  Additional local loss-function changes.

## Scope Of The Squashed Local Layer

The single local-layer commit on top of `1c601d4` is expected to cover the retained changes summarized above, rather than preserve the intermediate local-only commit history.
