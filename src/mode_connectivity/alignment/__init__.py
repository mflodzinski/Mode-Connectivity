"""Permutation-alignment primitives for model comparison workflows.

This package exposes permutation specifications, weight matching, and Sinkhorn
helpers used by the higher-level alignment and evaluation pipelines.
"""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / ".mplcache"))

from .permutation_spec import PermutationSpec, vgg16_permutation_spec
from .sinkhorn_utils import stable_sinkhorn
from .weight_matching import weight_matching, apply_permutation

__all__ = [
    'PermutationSpec',
    'vgg16_permutation_spec',
    'stable_sinkhorn',
    'weight_matching',
    'apply_permutation',
]
