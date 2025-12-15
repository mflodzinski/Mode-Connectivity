"""Model architecture utilities.

Provides functions for retrieving and instantiating model architectures.
Unified from scripts/analysis/lib/models.py and scripts/eval/lib/setup.py
"""

import sys
import torch
import torch.nn as nn
from typing import Any, Optional

# Add external dependencies to path
sys.path.insert(0, 'external/dnn-mode-connectivity')
import models as dnn_models


def get_architecture(model_name: str, use_bn: bool = False) -> Any:
    """Get model architecture object.

    Args:
        model_name: Base model name (e.g., 'VGG16', 'VGG19', 'ResNet18')
        use_bn: Whether to use batch normalization variant

    Returns:
        Architecture object with .base and .kwargs attributes
    """
    # Normalize model name
    model_name = model_name.upper().replace('-', '')

    # Add BN suffix if requested
    if use_bn:
        arch_name = f"{model_name}BN"
    else:
        arch_name = model_name

    # Get architecture from models module
    if not hasattr(dnn_models, arch_name):
        raise ValueError(f"Architecture '{arch_name}' not found in models module")

    return getattr(dnn_models, arch_name)


def create_model(architecture: Any,
                 num_classes: int = 10,
                 device: Optional[torch.device] = None,
                 **extra_kwargs) -> nn.Module:
    """Create model instance from architecture.

    Args:
        architecture: Architecture object (from get_architecture)
        num_classes: Number of output classes
        device: Optional device to place model on
        **extra_kwargs: Additional kwargs to pass to model constructor

    Returns:
        Model instance
    """
    # Handle architecture objects with .base and .kwargs
    if hasattr(architecture, 'base') and hasattr(architecture, 'kwargs'):
        base_class = architecture.base
        kwargs = {**architecture.kwargs, **extra_kwargs}
    else:
        base_class = architecture
        kwargs = extra_kwargs

    model = base_class(num_classes=num_classes, **kwargs)

    if device is not None:
        model.to(device)

    return model


def get_model(model_name: str,
              num_classes: int = 10,
              use_bn: bool = False,
              device: Optional[torch.device] = None,
              **extra_kwargs) -> nn.Module:
    """Get model instance by name.

    Convenience function combining get_architecture and create_model.

    Args:
        model_name: Base model name (e.g., 'VGG16', 'VGG19')
        num_classes: Number of output classes
        use_bn: Whether to use batch normalization variant
        device: Optional device to place model on
        **extra_kwargs: Additional kwargs to pass to model constructor

    Returns:
        Model instance
    """
    architecture = get_architecture(model_name, use_bn)
    return create_model(architecture, num_classes, device, **extra_kwargs)


def create_curve_model(architecture: Any,
                       num_classes: int,
                       curve_type: str,
                       num_bends: int,
                       device: Optional[torch.device] = None) -> nn.Module:
    """Create curve model (Bezier, PolyChain, etc.).

    Args:
        architecture: Architecture object from get_architecture()
        num_classes: Number of output classes
        curve_type: Curve type (e.g., 'Bezier', 'PolyChain')
        num_bends: Number of bend points
        device: Optional device to place model on

    Returns:
        CurveNet instance
    """
    # Import curves module from external repo
    import curves as external_curves

    curve_cls = getattr(external_curves, curve_type)

    model = external_curves.CurveNet(
        num_classes,
        curve_cls,
        architecture.curve,
        num_bends,
        architecture_kwargs=architecture.kwargs,
    )

    if device is not None:
        model.to(device)

    return model
