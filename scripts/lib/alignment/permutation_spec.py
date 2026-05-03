"""
Permutation specification for neural network weight matching.

Adapted from git-rebasin for PyTorch models.
"""

from collections import defaultdict
from typing import NamedTuple, Dict, List, Tuple, Optional


class PermutationSpec(NamedTuple):
    """Specification of how permutations affect model parameters.

    Attributes:
        perm_to_axes: Maps permutation name -> list of (param_key, axis) tuples
                      that this permutation affects.
        axes_to_perm: Maps param_key -> tuple of permutation names for each axis.
                      None means that axis is not permuted.
    """
    perm_to_axes: Dict[str, List[Tuple[str, int]]]
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]]


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    """Create PermutationSpec from axes_to_perm dictionary.

    Args:
        axes_to_perm: Dict mapping param keys to tuples of permutation names.

    Returns:
        PermutationSpec with both mappings populated.
    """
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def vgg16_permutation_spec() -> PermutationSpec:
    """Create permutation spec for VGG16 (PyTorch, no batch norm).

    PyTorch VGG16 structure:
    - layer_blocks: 5 blocks with [2, 2, 3, 3, 3] conv layers
    - classifier: Sequential with Dropout, Linear, ReLU, Dropout, Linear, ReLU, Linear

    PyTorch conv weight shape: [out_channels, in_channels, H, W]
    PyTorch linear weight shape: [out_features, in_features]

    State dict keys:
    - layer_blocks.{block}.{layer}.weight/bias
    - classifier.{idx}.weight/bias (idx: 1=FC1, 4=FC2, 6=FC3)
    """
    # Map (block_idx, layer_idx) to conv layer index (0-12)
    conv_layout = [
        (0, 0), (0, 1),           # Block 0: layers 0,1 -> conv 0,1
        (1, 0), (1, 1),           # Block 1: layers 0,1 -> conv 2,3
        (2, 0), (2, 1), (2, 2),   # Block 2: layers 0,1,2 -> conv 4,5,6
        (3, 0), (3, 1), (3, 2),   # Block 3: layers 0,1,2 -> conv 7,8,9
        (4, 0), (4, 1), (4, 2),   # Block 4: layers 0,1,2 -> conv 10,11,12
    ]

    def conv_key(block, layer):
        return f"layer_blocks.{block}.{layer}"

    axes_to_perm = {}

    # Conv layers
    # Conv 0: input from image (no input perm), output permuted
    block, layer = conv_layout[0]
    axes_to_perm[f"{conv_key(block, layer)}.weight"] = (None, None, None, "P_Conv_0")
    axes_to_perm[f"{conv_key(block, layer)}.bias"] = ("P_Conv_0",)

    # Conv 1-12: input from previous conv, output permuted
    for i in range(1, 13):
        block, layer = conv_layout[i]
        prev_block, prev_layer = conv_layout[i - 1]
        axes_to_perm[f"{conv_key(block, layer)}.weight"] = (
            f"P_Conv_{i}",      # output channels (axis 0 in PyTorch)
            f"P_Conv_{i-1}",    # input channels (axis 1 in PyTorch)
            None,               # kernel height
            None                # kernel width
        )
        axes_to_perm[f"{conv_key(block, layer)}.bias"] = (f"P_Conv_{i}",)

    # Fix Conv 0 weight axes (PyTorch: [out, in, H, W])
    block, layer = conv_layout[0]
    axes_to_perm[f"{conv_key(block, layer)}.weight"] = (
        "P_Conv_0",  # output channels (axis 0)
        None,        # input channels (3, fixed)
        None,        # kernel height
        None         # kernel width
    )

    # FC layers in classifier
    # classifier.1 = Linear(512, 512): input from last conv (flattened), output permuted
    axes_to_perm["classifier.1.weight"] = ("P_Dense_0", "P_Conv_12")
    axes_to_perm["classifier.1.bias"] = ("P_Dense_0",)

    # classifier.4 = Linear(512, 512): input from FC1, output permuted
    axes_to_perm["classifier.4.weight"] = ("P_Dense_1", "P_Dense_0")
    axes_to_perm["classifier.4.bias"] = ("P_Dense_1",)

    # classifier.6 = Linear(512, 10): input from FC2, output NOT permuted (class labels)
    axes_to_perm["classifier.6.weight"] = (None, "P_Dense_1")
    axes_to_perm["classifier.6.bias"] = (None,)

    return permutation_spec_from_axes_to_perm(axes_to_perm)


def vgg16_features_permutation_spec() -> PermutationSpec:
    """Create permutation spec for VGG16 with ``features.*``/``classifier.*`` keys.

    This matches the VGG16 layout used by:
    - ``external/pytorch-vgg-cifar10``
    - ``external/sinkhorn-rebasin/examples/models/vgg.py``
    """

    conv_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

    axes_to_perm = {}

    first_conv = conv_indices[0]
    axes_to_perm[f"features.{first_conv}.weight"] = ("P_Conv_0", None, None, None)
    axes_to_perm[f"features.{first_conv}.bias"] = ("P_Conv_0",)

    for i in range(1, len(conv_indices)):
        curr = conv_indices[i]
        axes_to_perm[f"features.{curr}.weight"] = (
            f"P_Conv_{i}",
            f"P_Conv_{i-1}",
            None,
            None,
        )
        axes_to_perm[f"features.{curr}.bias"] = (f"P_Conv_{i}",)

    axes_to_perm["classifier.1.weight"] = ("P_Dense_0", "P_Conv_12")
    axes_to_perm["classifier.1.bias"] = ("P_Dense_0",)
    axes_to_perm["classifier.4.weight"] = ("P_Dense_1", "P_Dense_0")
    axes_to_perm["classifier.4.bias"] = ("P_Dense_1",)
    axes_to_perm["classifier.6.weight"] = (None, "P_Dense_1")
    axes_to_perm["classifier.6.bias"] = (None,)

    return permutation_spec_from_axes_to_perm(axes_to_perm)


def open_lth_cifar_vgg16_bn_permutation_spec() -> PermutationSpec:
    """Permutation spec for open_lth CIFAR VGG16 with BatchNorm.

    State dict keys follow the pattern:
    - layers.{idx}.conv.weight / bias
    - layers.{idx}.bn.weight / bias / running_mean / running_var
    - fc.weight / bias

    The parameterized convolutional modules live at indices:
    [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
    """

    conv_indices = [0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
    axes_to_perm = {}

    first_conv = conv_indices[0]
    axes_to_perm[f"layers.{first_conv}.conv.weight"] = ("P_Conv_0", None, None, None)
    axes_to_perm[f"layers.{first_conv}.conv.bias"] = ("P_Conv_0",)

    for stat_name in ["weight", "bias", "running_mean", "running_var"]:
        axes_to_perm[f"layers.{first_conv}.bn.{stat_name}"] = ("P_Conv_0",)

    for i in range(1, len(conv_indices)):
        curr = conv_indices[i]
        axes_to_perm[f"layers.{curr}.conv.weight"] = (
            f"P_Conv_{i}",
            f"P_Conv_{i-1}",
            None,
            None,
        )
        axes_to_perm[f"layers.{curr}.conv.bias"] = (f"P_Conv_{i}",)
        for stat_name in ["weight", "bias", "running_mean", "running_var"]:
            axes_to_perm[f"layers.{curr}.bn.{stat_name}"] = (f"P_Conv_{i}",)

    axes_to_perm["fc.weight"] = (None, "P_Conv_12")
    axes_to_perm["fc.bias"] = (None,)
    return permutation_spec_from_axes_to_perm(axes_to_perm)


def mlp_permutation_spec(num_hidden_layers: int) -> PermutationSpec:
    """Create permutation spec for MLP.

    Args:
        num_hidden_layers: Number of hidden layers

    Returns:
        PermutationSpec for MLP with given depth
    """
    assert num_hidden_layers >= 1

    axes_to_perm = {
        "Dense_0.weight": (f"P_0", None),
        "Dense_0.bias": (f"P_0",),
    }

    for i in range(1, num_hidden_layers):
        axes_to_perm[f"Dense_{i}.weight"] = (f"P_{i}", f"P_{i-1}")
        axes_to_perm[f"Dense_{i}.bias"] = (f"P_{i}",)

    # Output layer
    axes_to_perm[f"Dense_{num_hidden_layers}.weight"] = (None, f"P_{num_hidden_layers-1}")
    axes_to_perm[f"Dense_{num_hidden_layers}.bias"] = (None,)

    return permutation_spec_from_axes_to_perm(axes_to_perm)
