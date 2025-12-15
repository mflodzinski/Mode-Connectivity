"""Setup utilities for evaluation scripts.

Handles initialization boilerplate:
- Device detection and selection
- External path configuration
"""

import sys
import os
import torch
from typing import Tuple, Dict, Any


def add_external_path():
    """Add external/dnn-mode-connectivity to Python path.

    This must be called before importing data, models, curves, utils.
    """
    external_path = os.path.join(
        os.path.dirname(__file__),
        '../../../external/dnn-mode-connectivity'
    )
    external_path = os.path.abspath(external_path)

    if external_path not in sys.path:
        sys.path.insert(0, external_path)


def get_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
    """Get best available device.

    Args:
        prefer_cuda: Prefer CUDA over MPS if available
        prefer_mps: Prefer MPS over CPU if available (Mac M1/M2)

    Returns:
        torch.device object
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    elif prefer_mps and torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"Using device: MPS (Apple Silicon)")
    else:
        device = torch.device('cpu')
        print(f"Using device: CPU")

    return device


# Backward compatibility wrapper class for eval scripts
class EvalSetup:
    """Backward compatibility wrapper providing old EvalSetup class interface.

    This class delegates to the appropriate modules (data, models, setup functions).
    """

    @staticmethod
    def add_external_path():
        """Add external/dnn-mode-connectivity to Python path."""
        return add_external_path()

    @staticmethod
    def get_device(prefer_cuda: bool = True, prefer_mps: bool = True) -> torch.device:
        """Get best available device."""
        return get_device(prefer_cuda, prefer_mps)

    @staticmethod
    def load_data(dataset: str,
                  data_path: str,
                  batch_size: int,
                  num_workers: int,
                  transform: str,
                  use_test: bool,
                  shuffle_train: bool = False) -> Tuple[Dict, int]:
        """Load dataset using data module."""
        from . import data
        return data.get_loaders(
            dataset=dataset,
            data_path=data_path,
            batch_size=batch_size,
            num_workers=num_workers,
            transform_name=transform,
            use_test=use_test,
            shuffle_train=shuffle_train
        )

    @staticmethod
    def get_architecture(model_name: str) -> Any:
        """Get model architecture from models module."""
        from . import models
        return models.get_architecture(model_name)

    @staticmethod
    def create_standard_model(architecture: Any,
                              num_classes: int,
                              device: torch.device) -> torch.nn.Module:
        """Create standard (non-curve) model."""
        from . import models
        return models.create_model(architecture, num_classes, device)

    @staticmethod
    def create_curve_model(architecture: Any,
                          num_classes: int,
                          curve_type: str,
                          num_bends: int,
                          device: torch.device) -> torch.nn.Module:
        """Create curve model (Bezier, PolyChain, etc.)."""
        from . import models
        return models.create_curve_model(architecture, num_classes, curve_type, num_bends, device)
