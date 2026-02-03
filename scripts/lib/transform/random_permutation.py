"""
Random permutation utilities for VGG16 networks.

Generates and applies full random layer-wise permutations to VGG16 models,
used for benchmarking permutation alignment methods.
"""

import torch
import numpy as np
from typing import Dict, OrderedDict
from copy import deepcopy


class VGG16RandomPermutation:
    """Generate and apply random layer-wise permutations to VGG16.

    VGG16 structure (no batch norm):
    - 5 conv blocks with [2, 2, 3, 3, 3] = 13 conv layers
    - 3 FC layers (2 hidden + 1 output)

    We permute output channels of each conv layer (affects next layer's input)
    and output neurons of hidden FC layers (affects next layer's input).
    The final output layer is NOT permuted (class labels are fixed).
    """

    # VGG16 layer structure: (block_idx, layer_idx) -> output channels
    CONV_LAYERS = [
        (0, 0, 64), (0, 1, 64),      # Block 0: 2 layers, 64 channels
        (1, 0, 128), (1, 1, 128),    # Block 1: 2 layers, 128 channels
        (2, 0, 256), (2, 1, 256), (2, 2, 256),  # Block 2: 3 layers, 256 channels
        (3, 0, 512), (3, 1, 512), (3, 2, 512),  # Block 3: 3 layers, 512 channels
        (4, 0, 512), (4, 1, 512), (4, 2, 512),  # Block 4: 3 layers, 512 channels
    ]

    # FC layer indices in classifier Sequential (after Dropout/ReLU)
    # classifier = [Dropout, Linear(512,512), ReLU, Dropout, Linear(512,512), ReLU, Linear(512,10)]
    FC_LAYERS = [
        (1, 512),  # classifier.1: Linear(512, 512)
        (4, 512),  # classifier.4: Linear(512, 512)
        # classifier.6 is output layer - not permuted
    ]

    def __init__(self):
        pass

    def generate(self, seed: int = 42) -> Dict[str, np.ndarray]:
        """Generate random permutation for each permutable layer.

        Args:
            seed: Random seed for reproducibility

        Returns:
            Dictionary mapping layer names to permutation index arrays.
            Keys: 'conv_0' through 'conv_12', 'fc_0', 'fc_1'
        """
        rng = np.random.RandomState(seed)
        perm = {}

        # Generate permutations for conv layers
        for i, (block_idx, layer_idx, channels) in enumerate(self.CONV_LAYERS):
            perm[f'conv_{i}'] = rng.permutation(channels).astype(np.int64)

        # Generate permutations for hidden FC layers
        for i, (fc_idx, neurons) in enumerate(self.FC_LAYERS):
            perm[f'fc_{i}'] = rng.permutation(neurons).astype(np.int64)

        return perm

    def invert(self, perm: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get inverse permutation.

        If P is a permutation, P_inv satisfies: x[P][P_inv] = x

        Args:
            perm: Original permutation dictionary

        Returns:
            Inverse permutation dictionary
        """
        perm_inv = {}
        for key, p in perm.items():
            # Inverse permutation: if P[i] = j, then P_inv[j] = i
            p_inv = np.zeros_like(p)
            p_inv[p] = np.arange(len(p))
            perm_inv[key] = p_inv
        return perm_inv

    def apply_to_state_dict(
        self,
        state_dict: OrderedDict,
        perm: Dict[str, np.ndarray]
    ) -> OrderedDict:
        """Apply permutation to a VGG16 state_dict.

        This permutes:
        1. Output channels of each conv layer (and its bias)
        2. Input channels of the next layer
        3. For conv->FC transition: handle flattening

        Args:
            state_dict: PyTorch state dict (will NOT be modified)
            perm: Permutation dictionary from generate()

        Returns:
            New state dict with permutation applied
        """
        # Deep copy to avoid modifying original
        new_state = OrderedDict()
        for k, v in state_dict.items():
            new_state[k] = v.clone()

        # Apply conv layer permutations
        for i in range(13):  # 13 conv layers
            p_out = torch.tensor(perm[f'conv_{i}'], dtype=torch.long)

            # Get layer location
            block_idx, layer_idx, _ = self.CONV_LAYERS[i]
            weight_key = f'layer_blocks.{block_idx}.{layer_idx}.weight'
            bias_key = f'layer_blocks.{block_idx}.{layer_idx}.bias'

            # Permute output channels: weight shape is [out, in, H, W]
            new_state[weight_key] = new_state[weight_key][p_out]
            if bias_key in new_state:
                new_state[bias_key] = new_state[bias_key][p_out]

            # Permute input channels of next layer
            if i < 12:  # Not the last conv layer
                next_block_idx, next_layer_idx, _ = self.CONV_LAYERS[i + 1]
                next_weight_key = f'layer_blocks.{next_block_idx}.{next_layer_idx}.weight'
                # Permute input channels: weight[:, p_out, :, :]
                new_state[next_weight_key] = new_state[next_weight_key][:, p_out, :, :]
            else:
                # Last conv layer -> first FC layer (with flattening)
                # Conv output is [512, 1, 1] after pooling, flattened to 512
                # FC weight shape is [out_features, in_features] = [512, 512]
                fc_weight_key = 'classifier.1.weight'
                # Since spatial size is 1x1, flattening is trivial
                # Just permute input features directly
                new_state[fc_weight_key] = new_state[fc_weight_key][:, p_out]

        # Apply FC layer permutations
        for i, (fc_idx, _) in enumerate(self.FC_LAYERS):
            p_out = torch.tensor(perm[f'fc_{i}'], dtype=torch.long)

            weight_key = f'classifier.{fc_idx}.weight'
            bias_key = f'classifier.{fc_idx}.bias'

            # Permute output neurons
            new_state[weight_key] = new_state[weight_key][p_out]
            if bias_key in new_state:
                new_state[bias_key] = new_state[bias_key][p_out]

            # Permute input of next FC layer
            if i == 0:
                # FC1 -> FC2
                next_weight_key = 'classifier.4.weight'
            else:
                # FC2 -> FC3 (output layer)
                next_weight_key = 'classifier.6.weight'
            new_state[next_weight_key] = new_state[next_weight_key][:, p_out]

        return new_state

    def verify_equivalence(
        self,
        model_original,
        model_permuted,
        test_input: torch.Tensor
    ) -> bool:
        """Verify that original and permuted models produce same output.

        This should return True if permutation was applied correctly.

        Args:
            model_original: Original model
            model_permuted: Model with permutation applied
            test_input: Test input tensor [batch, 3, 32, 32]

        Returns:
            True if outputs match within tolerance
        """
        model_original.eval()
        model_permuted.eval()

        with torch.no_grad():
            out_orig = model_original(test_input)
            out_perm = model_permuted(test_input)

        max_diff = (out_orig - out_perm).abs().max().item()
        return max_diff < 1e-5


def test_permutation():
    """Test that applying P then P_inv recovers original state_dict."""
    import sys
    sys.path.insert(0, 'external/dnn-mode-connectivity')
    import models

    # Create model
    model = models.VGG16.base(num_classes=10)
    original_state = deepcopy(model.state_dict())

    # Generate permutation
    perm_gen = VGG16RandomPermutation()
    P = perm_gen.generate(seed=42)
    P_inv = perm_gen.invert(P)

    # Apply P then P_inv
    permuted_state = perm_gen.apply_to_state_dict(original_state, P)
    recovered_state = perm_gen.apply_to_state_dict(permuted_state, P_inv)

    # Check equality
    all_close = True
    for key in original_state:
        if not torch.allclose(original_state[key], recovered_state[key], atol=1e-6):
            print(f"Mismatch in {key}")
            all_close = False

    if all_close:
        print("SUCCESS: P_inv(P(state)) == state")
    else:
        print("FAILURE: State not recovered")

    return all_close


if __name__ == '__main__':
    test_permutation()
