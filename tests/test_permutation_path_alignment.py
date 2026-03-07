import os
import sys
import unittest
from collections import OrderedDict

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from scripts.lib.alignment.c2m3_bridge import local_spec_to_c2m3
from scripts.lib.alignment.path_checkpoint_sampling import curve_state_dict_at_t, state_dicts_allclose
from scripts.lib.alignment.permutation_pipeline import (
    apply_endpoint_permutation_to_state_dict,
    compose_permutations,
    convert_apply_keys_to_perm_format,
    convert_perm_keys_to_apply_format,
    derive_endpoint_permutation_from_factored,
    unfactor_factored_permutations,
)
from scripts.lib.alignment.permutation_spec import vgg16_permutation_spec
from scripts.lib.analysis.alignment import create_vgg16_model
from scripts.lib.core.models import create_curve_model, create_model, get_architecture
from scripts.lib.core.setup import add_external_path
from scripts.lib.transform.random_permutation import VGG16RandomPermutation


class PermutationPathAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        add_external_path()
        cls.local_spec = vgg16_permutation_spec()
        cls.architecture = get_architecture("VGG16")

    def test_local_spec_translation_preserves_keys_and_sizes(self):
        c2m3_spec = local_spec_to_c2m3(self.local_spec)
        self.assertEqual(set(self.local_spec.axes_to_perm.keys()), set(c2m3_spec.layer_and_axes_to_perm.keys()))

        model = create_vgg16_model(num_classes=10, device=torch.device("cpu"))
        state_dict = model.state_dict()

        for perm_name, axes in self.local_spec.perm_to_axes.items():
            local_param_name, local_axis = axes[0]
            c2m3_param_name, c2m3_axis = c2m3_spec.perm_to_layers_and_axes[perm_name][0]
            self.assertEqual(local_param_name, c2m3_param_name)
            self.assertEqual(local_axis, c2m3_axis)
            self.assertEqual(
                state_dict[local_param_name].shape[local_axis],
                state_dict[c2m3_param_name].shape[c2m3_axis],
            )

    def test_permutation_composition_matches_sequential_state_application(self):
        model = create_vgg16_model(num_classes=10, device=torch.device("cpu"))
        state = OrderedDict((key, value.clone()) for key, value in model.state_dict().items())

        perm_gen = VGG16RandomPermutation()
        right_apply = perm_gen.generate(seed=7)
        left_apply = perm_gen.generate(seed=11)
        right = convert_apply_keys_to_perm_format(right_apply)
        left = convert_apply_keys_to_perm_format(left_apply)

        sequential = perm_gen.apply_to_state_dict(
            perm_gen.apply_to_state_dict(OrderedDict((key, value.clone()) for key, value in state.items()), right_apply),
            left_apply,
        )
        composed = compose_permutations(left, right)
        composed_apply = convert_perm_keys_to_apply_format(composed)
        combined = perm_gen.apply_to_state_dict(state, composed_apply)

        self.assertTrue(state_dicts_allclose(sequential, combined))

    def test_factored_endpoint_derivation_matches_unfactoring(self):
        perm_gen = VGG16RandomPermutation()
        factored = {
            "C0": convert_apply_keys_to_perm_format(perm_gen.generate(seed=0)),
            "C1": convert_apply_keys_to_perm_format(perm_gen.generate(seed=1)),
            "C4": convert_apply_keys_to_perm_format(perm_gen.generate(seed=4)),
        }

        unfactored = unfactor_factored_permutations(factored)
        derived = derive_endpoint_permutation_from_factored(
            factored,
            fixed_symbol="C0",
            permutee_symbol="C4",
        )

        for perm_name, indices in derived.items():
            self.assertTrue((indices == unfactored["C0"]["C4"][perm_name]).all())

    def test_curve_sampling_reconstructs_path_endpoints(self):
        curve_model = create_curve_model(
            self.architecture,
            num_classes=10,
            curve_type="PolyChain",
            num_bends=3,
            device=torch.device("cpu"),
        )
        model_a = create_model(self.architecture, num_classes=10, device=torch.device("cpu"))
        model_b = create_model(self.architecture, num_classes=10, device=torch.device("cpu"))

        curve_model.import_base_parameters(model_a, 0)
        curve_model.import_base_parameters(model_b, 2)
        curve_model.init_linear()

        sampled_a = curve_state_dict_at_t(curve_model, t=0.0, model_name="VGG16", num_classes=10, device="cpu")
        sampled_b = curve_state_dict_at_t(curve_model, t=1.0, model_name="VGG16", num_classes=10, device="cpu")

        self.assertTrue(state_dicts_allclose(sampled_a, model_a.state_dict()))
        self.assertTrue(state_dicts_allclose(sampled_b, model_b.state_dict()))

    def test_applying_endpoint_permutation_preserves_logits(self):
        model = create_vgg16_model(num_classes=10, device=torch.device("cpu"))
        state = OrderedDict((key, value.clone()) for key, value in model.state_dict().items())

        perm_gen = VGG16RandomPermutation()
        endpoint_q = convert_apply_keys_to_perm_format(perm_gen.generate(seed=123))
        permuted_state = apply_endpoint_permutation_to_state_dict(state, endpoint_q)

        model_original = create_vgg16_model(num_classes=10, device=torch.device("cpu"))
        model_permuted = create_vgg16_model(num_classes=10, device=torch.device("cpu"))
        model_original.load_state_dict(state)
        model_permuted.load_state_dict(permuted_state)
        model_original.eval()
        model_permuted.eval()

        inputs = torch.randn(8, 3, 32, 32)
        with torch.no_grad():
            outputs_original = model_original(inputs)
            outputs_permuted = model_permuted(inputs)

        self.assertTrue(torch.allclose(outputs_original, outputs_permuted, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
