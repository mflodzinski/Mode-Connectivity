"""Unified library for mode connectivity analysis and evaluation.

Provides reusable modules organized by functionality:
- core: Essential components (checkpoints, data, models, setup, I/O)
- evaluation: Model evaluation and metrics
- curves: Curve-specific utilities and analysis
- transform: Network transformations (permutation, mirroring, neuron swapping)
- analysis: Analysis utilities (plotting, prediction analysis)
- utils: Command-line argument parsing
"""

from .core import checkpoint, data, models, setup, output
from .evaluation import metrics, interpolation, evaluate
from .curves import curves, analyzer
from .transform import permutation, mirror, neuron_swap
from .analysis import plotting, prediction_analyzer
from .utils import args

__all__ = [
    # Core
    'checkpoint',
    'data',
    'models',
    'setup',
    'output',
    # Evaluation
    'metrics',
    'interpolation',
    'evaluate',
    # Curves
    'curves',
    'analyzer',
    # Transform
    'permutation',
    'mirror',
    'neuron_swap',
    # Analysis
    'plotting',
    'prediction_analyzer',
    # Utils
    'args',
]
