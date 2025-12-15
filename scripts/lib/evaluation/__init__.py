"""Evaluation utilities for mode connectivity experiments.

Provides:
- metrics: Distance and similarity metrics between models
- interpolation: Weight interpolation along paths
- evaluate: Model evaluation on datasets and along paths
"""

from . import metrics
from . import interpolation
from . import evaluate

__all__ = [
    'metrics',
    'interpolation',
    'evaluate',
]
