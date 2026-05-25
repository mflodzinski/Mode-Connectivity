# Vendor Changes For Mode-Connectivity

This vendored directory was imported from an external source tree. At the time this note was added, that source tree was clean and pinned at commit `79518a3` on `master`.

## Comparison Basis

This vendored directory did not have a separate `upstream` remote configured locally before it was imported. In the visible local history, the original project lineage ends at commit `f0bf253`, and all later commits are local changes for this repository. Those local changes are intended to be kept as a single squashed layer on top of that base.

## What Changed Locally

- The training surface in `train.py` has been extended well beyond the original curve-training CLI.
  Current local options include:
  `--no_train_aug`, `--train_half_only`, advanced initialization methods, W&B logging, early stopping, train/val/test splitting, symmetry-plane projection, and random-plane projection.
- The dataset loader in `data.py` has been adapted to support evaluation-mode loading without train augmentation and to behave better on Apple MPS by forcing `num_workers=0` and disabling `pin_memory` there.
- Curve training can now optimize only half of the path by setting `train_t_min` / `train_t_max`; that behavior is wired through `train.py` and consumed in `curves.py`.
- Several evaluation and analysis entrypoints were updated together with transform handling corrections:
  `connect.py`, `eval_curve.py`, `eval_ensemble.py`, `plane.py`, and `test_curve.py`.
- Earlier local MNIST / FashionMNIST support was later removed again. The current pinned state is back to a CIFAR-focused loader surface, and that cleanup happened in `data.py`.

## Main Places To Inspect

- `train.py`
  Main local surface. Most added flags and orchestration changes live here.
- `data.py`
  Dataset loading, transform selection, MPS behavior, and the removal of earlier MNIST/FashionMNIST additions.
- `curves.py`
  Runtime support for half-path training through `train_t_min` / `train_t_max`.
- `connect.py`, `eval_curve.py`, `eval_ensemble.py`, `plane.py`, `test_curve.py`
  Small coordinated fixes around transform handling and evaluation behavior.

## Scope Of The Squashed Local Layer

The single local-layer commit on top of `f0bf253` is expected to cover the retained changes summarized above, rather than preserve the intermediate local-only commit history.
