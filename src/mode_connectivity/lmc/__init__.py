"""Reusable helpers for linear mode-connectivity benchmark workflows.

The package currently exposes the shared-training implementation that powers the
repo-level LMC experiment runners.
"""

from . import shared_training

__all__ = ["shared_training"]
