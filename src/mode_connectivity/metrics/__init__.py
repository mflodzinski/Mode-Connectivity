"""Convenience exports for metric helpers.

This package currently re-exports the canonical checkpoint-distance functions
from the main evaluation metrics module.
"""

from .distances import (
    calculate_checkpoint_l2_distance,
    print_l2_statistics,
    save_l2_distance_report,
)

__all__ = [
    "calculate_checkpoint_l2_distance",
    "print_l2_statistics",
    "save_l2_distance_report",
]
