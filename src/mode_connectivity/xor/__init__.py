"""Reusable XOR experiment implementations.

This package intentionally avoids eager submodule imports because some retained
XOR modules pull optional plotting dependencies such as ``plotly``. The thin
repo-level runners import only the concrete module they need.
"""

__all__ = [
    "xor_curve_fitting",
    "xor_experiment",
    "xor_permutation_scale_experiment",
    "xor_train_linear_barriers",
]
