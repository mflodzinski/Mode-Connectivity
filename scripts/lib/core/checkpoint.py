"""Checkpoint and model loading utilities.

Provides unified interfaces for loading checkpoints and models.
Combines functionality from:
- scripts/analysis/lib/checkpoint.py
- scripts/eval/lib/checkpoint_loader.py
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any, Union


# ============================================================================
# Functional API (from analysis/lib/checkpoint.py)
# ============================================================================

def load_checkpoint(path: str, map_location: str = 'cpu') -> Dict[str, Any]:
    """Load a checkpoint file.

    Args:
        path: Path to checkpoint file
        map_location: Device to map tensors to (default: 'cpu')

    Returns:
        Checkpoint dictionary
    """
    return torch.load(path, map_location=map_location)


def load_state_dict(path: str, map_location: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Load state dict from checkpoint.

    Handles both direct state dicts and checkpoints with 'model_state' key.

    Args:
        path: Path to checkpoint file
        map_location: Device to map tensors to (default: 'cpu')

    Returns:
        Model state dictionary
    """
    checkpoint = load_checkpoint(path, map_location)

    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        return checkpoint['model_state']

    return checkpoint


def load_model(checkpoint_path: str, model_class: type, num_classes: int = 10,
               model_kwargs: Optional[Dict] = None, map_location: str = 'cpu') -> nn.Module:
    """Load a model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_class: Model class or architecture object with .base attribute
        num_classes: Number of output classes
        model_kwargs: Additional kwargs for model initialization
        map_location: Device to map tensors to

    Returns:
        Loaded model with weights
    """
    # Handle architecture objects with .base and .kwargs attributes
    if hasattr(model_class, 'base') and hasattr(model_class, 'kwargs'):
        base_class = model_class.base
        kwargs = {**model_class.kwargs}
        if model_kwargs:
            kwargs.update(model_kwargs)
    else:
        base_class = model_class
        kwargs = model_kwargs or {}

    # Create model
    model = base_class(num_classes=num_classes, **kwargs)

    # Load weights
    state_dict = load_state_dict(checkpoint_path, map_location)
    model.load_state_dict(state_dict)

    return model


def load_model_into(model: nn.Module, checkpoint_path: str, map_location: str = 'cpu') -> None:
    """Load checkpoint weights into an existing model.

    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        map_location: Device to map tensors to
    """
    state_dict = load_state_dict(checkpoint_path, map_location)
    model.load_state_dict(state_dict)


# ============================================================================
# Class-based API (from eval/lib/checkpoint_loader.py)
# ============================================================================

class CheckpointLoader:
    """Load and manage checkpoints with device-aware consistent interface."""

    def __init__(self, device: torch.device):
        """Initialize checkpoint loader.

        Args:
            device: Device to load checkpoints onto
        """
        self.device = device

    def load_single(self, path: str) -> Dict[str, Any]:
        """Load single checkpoint.

        Args:
            path: Path to checkpoint file

        Returns:
            Checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=self.device)
        return checkpoint

    def load_endpoints(self, path_start: str, path_end: str) -> Tuple[Dict, Dict]:
        """Load two endpoint checkpoints.

        Args:
            path_start: Path to first endpoint checkpoint
            path_end: Path to second endpoint checkpoint

        Returns:
            Tuple of (state_dict_start, state_dict_end)
        """
        ckpt_start = torch.load(path_start, map_location=self.device)
        ckpt_end = torch.load(path_end, map_location=self.device)

        # Extract state dicts
        weights_start = self.get_state_dict(ckpt_start)
        weights_end = self.get_state_dict(ckpt_end)

        return weights_start, weights_end

    def load_symmetry(self,
                      path_start: str,
                      path_theta: str,
                      path_end: str) -> Tuple[Dict, Dict, Dict]:
        """Load three checkpoints for symmetry plane evaluation.

        Args:
            path_start: Path to first endpoint checkpoint (w1)
            path_theta: Path to symmetry point checkpoint (θ*)
            path_end: Path to second endpoint checkpoint (w2)

        Returns:
            Tuple of (state_dict_start, state_dict_theta, state_dict_end)
        """
        ckpt1 = torch.load(path_start, map_location=self.device)
        ckpt_theta = torch.load(path_theta, map_location=self.device)
        ckpt2 = torch.load(path_end, map_location=self.device)

        # Extract state dicts
        w1 = self.get_state_dict(ckpt1)
        theta = self.get_state_dict(ckpt_theta)
        w2 = self.get_state_dict(ckpt2)

        return w1, theta, w2

    def load_curve_with_endpoints(self,
                                   curve_path: str,
                                   endpoint0_path: str,
                                   endpoint1_path: str) -> Tuple[Dict, Dict, Dict]:
        """Load curve checkpoint plus both endpoints.

        Used for detailed analysis where we need both curve and endpoint models.

        Args:
            curve_path: Path to curve checkpoint
            endpoint0_path: Path to first endpoint checkpoint
            endpoint1_path: Path to second endpoint checkpoint

        Returns:
            Tuple of (curve_checkpoint, endpoint0_state, endpoint1_state)
        """
        curve_checkpoint = torch.load(curve_path, map_location=self.device)
        endpoint0_checkpoint = torch.load(endpoint0_path, map_location=self.device)
        endpoint1_checkpoint = torch.load(endpoint1_path, map_location=self.device)

        # For endpoints, extract state dicts
        endpoint0_state = self.get_state_dict(endpoint0_checkpoint)
        endpoint1_state = self.get_state_dict(endpoint1_checkpoint)

        # For curve, return full checkpoint (may contain curve-specific metadata)
        return curve_checkpoint, endpoint0_state, endpoint1_state

    @staticmethod
    def get_state_dict(checkpoint: Union[Dict, Any]) -> Dict[str, torch.Tensor]:
        """Extract state dict from checkpoint.

        Handles two formats:
        1. Checkpoint dict with 'model_state' key
        2. Raw state dict

        Args:
            checkpoint: Checkpoint object (dict or state dict)

        Returns:
            State dictionary
        """
        if isinstance(checkpoint, dict):
            # Check if it's a checkpoint dict with 'model_state' key
            if 'model_state' in checkpoint:
                return checkpoint['model_state']
            # Otherwise assume it's already a state dict
            return checkpoint
        else:
            # If not a dict, try to convert
            return dict(checkpoint)

    def load_into_model(self, model: nn.Module, checkpoint_path: str):
        """Load checkpoint directly into model.

        Args:
            model: Model to load weights into
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = self.load_single(checkpoint_path)
        state_dict = self.get_state_dict(checkpoint)
        model.load_state_dict(state_dict)

    def load_weights_into_model(self, model: nn.Module, weights: Dict[str, torch.Tensor]):
        """Load state dict weights into model.

        Args:
            model: Model to load weights into
            weights: State dictionary
        """
        model.load_state_dict(weights)
