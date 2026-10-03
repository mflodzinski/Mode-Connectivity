"""Reusable helpers for Sinkhorn-based alignment workflows.

The package keeps shared VGG/CIFAR alignment utilities separate from the
repo-level experiment runners that call into them.
"""

from . import shared
from . import sweep_utils

__all__ = ["shared", "sweep_utils"]
