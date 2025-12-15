"""Network transformation utilities.

Provides:
- permutation: Neuron permutation operations
- mirror: Creating mirrored networks via reverse permutation
- neuron_swap: Swapping specific neurons for minimal perturbations
"""

from . import permutation
from . import mirror
from . import neuron_swap

__all__ = [
    'permutation',
    'mirror',
    'neuron_swap',
]
