"""Checkpoint and model loading utilities.

This is the canonical place for decoding retained checkpoint formats, normalizing
state dicts, detecting supported families, and hydrating matching model objects.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from mode_connectivity.core import models as core_models
from mode_connectivity.external import load_vgg_module


StateDict = Dict[str, torch.Tensor]
CheckpointPayload = Union[Dict[str, Any], StateDict]


def _load_pytorch_vgg_module():
    return load_vgg_module()


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> CheckpointPayload:
    """Load a checkpoint payload from disk."""

    return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint: CheckpointPayload) -> StateDict:
    """Extract a state dict from a retained checkpoint payload."""

    if not isinstance(checkpoint, dict):
        return dict(checkpoint)

    if "model_state" in checkpoint:
        return checkpoint["model_state"]
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def normalize_state_dict_keys(state_dict: StateDict) -> OrderedDict[str, torch.Tensor]:
    """Normalize common wrapper prefixes without changing tensor values."""

    normalized = OrderedDict()
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        if normalized_key.startswith("features.module."):
            normalized_key = "features." + normalized_key[len("features.module.") :]
        normalized[normalized_key] = value
    return normalized


def load_state_dict(
    path: str | Path,
    map_location: str | torch.device = "cpu",
    *,
    normalize_keys: bool = False,
) -> StateDict:
    """Load a state dict from disk."""

    state_dict = extract_state_dict(load_checkpoint(path, map_location))
    if normalize_keys:
        return normalize_state_dict_keys(state_dict)
    return state_dict


def detect_vgg_checkpoint_family(state_dict: StateDict) -> str:
    """Detect the retained VGG checkpoint family from state-dict keys."""

    keys = list(state_dict.keys())
    if any(key.startswith("layer_blocks.") for key in keys):
        return "dnn_mode_connectivity"
    if any(key.startswith("features.") or key.startswith("features.module.") for key in keys):
        return "pytorch_vgg_cifar10"
    raise ValueError("Unsupported retained VGG checkpoint format.")


def load_checkpoint_state(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> Tuple[OrderedDict[str, torch.Tensor], str]:
    """Load a checkpoint, normalize keys, and detect its retained VGG family."""

    state_dict = normalize_state_dict_keys(load_state_dict(checkpoint_path, map_location))
    checkpoint_family = detect_vgg_checkpoint_family(state_dict)
    return state_dict, checkpoint_family


def build_model_for_vgg_checkpoint_family(
    checkpoint_family: str,
    *,
    num_classes: int = 10,
) -> nn.Module:
    """Instantiate the retained VGG architecture for a checkpoint family."""

    if checkpoint_family == "dnn_mode_connectivity":
        return core_models.get_model("VGG16", num_classes=num_classes)
    if checkpoint_family == "pytorch_vgg_cifar10":
        return _load_pytorch_vgg_module().vgg16()
    raise ValueError(f"Unsupported retained VGG checkpoint family: {checkpoint_family}")


def build_model_from_state_dict(
    state_dict: StateDict,
    *,
    checkpoint_family: Optional[str] = None,
    num_classes: int = 10,
    model_factory: Optional[Callable[[], nn.Module]] = None,
) -> nn.Module:
    """Instantiate and load a model from a normalized state dict."""

    normalized_state_dict = normalize_state_dict_keys(state_dict)
    if model_factory is not None:
        model = model_factory()
    else:
        family = checkpoint_family or detect_vgg_checkpoint_family(normalized_state_dict)
        model = build_model_for_vgg_checkpoint_family(family, num_classes=num_classes)
    model.load_state_dict(normalized_state_dict)
    return model


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    checkpoint_family: Optional[str] = None,
    num_classes: int = 10,
    model_factory: Optional[Callable[[], nn.Module]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[nn.Module, OrderedDict[str, torch.Tensor], str]:
    """Load a model plus its normalized state dict from a retained checkpoint."""

    state_dict, detected_family = load_checkpoint_state(checkpoint_path, map_location=map_location)
    family = checkpoint_family or detected_family
    model = build_model_from_state_dict(
        state_dict,
        checkpoint_family=family,
        num_classes=num_classes,
        model_factory=model_factory,
    )
    if device is not None:
        model = model.to(device)
    return model, state_dict, family


def load_model(
    checkpoint_path: str,
    model_class: type,
    num_classes: int = 10,
    model_kwargs: Optional[Dict] = None,
    map_location: str = "cpu",
) -> nn.Module:
    """Load a model from checkpoint using an explicit architecture class."""

    if hasattr(model_class, "base") and hasattr(model_class, "kwargs"):
        base_class = model_class.base
        kwargs = {**model_class.kwargs}
        if model_kwargs:
            kwargs.update(model_kwargs)
    else:
        base_class = model_class
        kwargs = model_kwargs or {}

    model = base_class(num_classes=num_classes, **kwargs)
    state_dict = load_state_dict(checkpoint_path, map_location)
    model.load_state_dict(state_dict)
    return model


def load_model_into(model: nn.Module, checkpoint_path: str, map_location: str = "cpu") -> None:
    """Load checkpoint weights into an existing model."""

    state_dict = load_state_dict(checkpoint_path, map_location)
    model.load_state_dict(state_dict)


class CheckpointLoader:
    """Load and manage checkpoints with a device-aware interface."""

    def __init__(self, device: torch.device):
        self.device = device

    def load_single(self, path: str) -> CheckpointPayload:
        return load_checkpoint(path, map_location=self.device)

    def load_endpoints(self, path_start: str, path_end: str) -> Tuple[StateDict, StateDict]:
        ckpt_start = self.load_single(path_start)
        ckpt_end = self.load_single(path_end)
        return self.get_state_dict(ckpt_start), self.get_state_dict(ckpt_end)

    def load_symmetry(self, path_start: str, path_theta: str, path_end: str) -> Tuple[StateDict, StateDict, StateDict]:
        ckpt1 = self.load_single(path_start)
        ckpt_theta = self.load_single(path_theta)
        ckpt2 = self.load_single(path_end)
        return self.get_state_dict(ckpt1), self.get_state_dict(ckpt_theta), self.get_state_dict(ckpt2)

    def load_curve_with_endpoints(
        self,
        curve_path: str,
        endpoint0_path: str,
        endpoint1_path: str,
    ) -> Tuple[CheckpointPayload, StateDict, StateDict]:
        curve_checkpoint = self.load_single(curve_path)
        endpoint0_checkpoint = self.load_single(endpoint0_path)
        endpoint1_checkpoint = self.load_single(endpoint1_path)
        return curve_checkpoint, self.get_state_dict(endpoint0_checkpoint), self.get_state_dict(endpoint1_checkpoint)

    @staticmethod
    def get_state_dict(checkpoint: CheckpointPayload) -> StateDict:
        return extract_state_dict(checkpoint)

    def load_into_model(self, model: nn.Module, checkpoint_path: str):
        model.load_state_dict(self.get_state_dict(self.load_single(checkpoint_path)))

    def load_weights_into_model(self, model: nn.Module, weights: StateDict):
        model.load_state_dict(weights)
