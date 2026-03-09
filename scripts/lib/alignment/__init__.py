"""Permutation alignment algorithms for neural networks."""

import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(_PROJECT_ROOT / ".mplcache"))

from .permutation_spec import PermutationSpec, vgg16_permutation_spec
from .weight_matching import weight_matching, apply_permutation
from .vgg16_sinkhorn_alignment import (
    METHOD_PERM_ONLY,
    METHOD_PERM_SCALE,
    VGG16_HIDDEN_LAYER_SPECS,
    VGG16AlignmentParameters,
    apply_alignment_to_state_dict,
    build_hard_alignment_from_indices,
    build_identity_alignment,
    build_hard_alignment_from_soft,
    run_vgg16_alignment_experiment,
)
from .vgg16_sinkhorn_evaluation import run_vgg16_alignment_evaluation

__all__ = [
    'PermutationSpec',
    'vgg16_permutation_spec',
    'weight_matching',
    'apply_permutation',
    'METHOD_PERM_ONLY',
    'METHOD_PERM_SCALE',
    'VGG16_HIDDEN_LAYER_SPECS',
    'VGG16AlignmentParameters',
    'apply_alignment_to_state_dict',
    'build_hard_alignment_from_indices',
    'build_hard_alignment_from_soft',
    'build_identity_alignment',
    'run_vgg16_alignment_experiment',
    'run_vgg16_alignment_evaluation',
]
