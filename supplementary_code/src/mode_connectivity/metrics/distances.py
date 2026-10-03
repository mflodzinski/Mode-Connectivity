"""Compatibility re-exports for checkpoint distance helpers.

This file preserves a small legacy import surface while delegating the real
implementations to the canonical evaluation metrics module.
"""

from mode_connectivity.evaluation.metrics import (
    calculate_checkpoint_l2_distance,
    print_l2_statistics,
    save_l2_distance_report,
)

__all__ = [
    "calculate_checkpoint_l2_distance",
    "print_l2_statistics",
    "save_l2_distance_report",
]
