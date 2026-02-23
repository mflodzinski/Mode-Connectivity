"""Network transformation utilities.

Provides:
- permutation: Neuron permutation operations
- mirror: Creating mirrored networks via reverse permutation
- neuron_swap: Swapping specific neurons for minimal perturbations
- random_permutation: Full-network random permutations (VGG16)
"""

from . import permutation
from . import mirror
from . import neuron_swap
from . import random_permutation

__all__ = [
    'permutation',
    'mirror',
    'neuron_swap',
    'random_permutation',
]
