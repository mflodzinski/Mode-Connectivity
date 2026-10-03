# `mode_connectivity` library

The retained packages support the paper workflows:

- `xor/`: XOR endpoint, exhaustive-alignment, scale, and nonlinear-path implementations.
- `training_stage/`: independent VGG training and frozen-data protocol.
- `fashion_mnist/`: deep-MLP model, frozen-data protocol, and endpoint training.
- `dense_linear_stage/`: final larger-network alignment search, evaluation, and reporting. Its historical name is retained for checkpoint/config compatibility; cross-stage reporting and launch modes have been removed.
- `alignment/`: permutation specifications, weight matching, and stable Sinkhorn operations.
- `sinkhorn/shared.py`: shared scale-aware Sinkhorn transformations used by the VGG pipeline.
- `evaluation/`, `core/`, and `common/`: metrics, checkpoint/data utilities, config composition, and paths.
- `external/`: import adapters for vendored upstream code.

Repo-facing commands live in [`experiments/`](../../experiments/README.md); post-processing scripts live in [`tools/`](../../tools/README.md).
