"""Reusable XOR experiment implementations.

The modules in this package keep the actual XOR training, alignment, and path
evaluation logic behind the thin repo-level runners in ``experiments.xor``.
"""

from . import xor_curve_fitting
from . import xor_experiment
from . import xor_permutation_scale_experiment
from . import xor_train_linear_barriers

__all__ = [
    "xor_curve_fitting",
    "xor_experiment",
    "xor_permutation_scale_experiment",
    "xor_train_linear_barriers",
]
