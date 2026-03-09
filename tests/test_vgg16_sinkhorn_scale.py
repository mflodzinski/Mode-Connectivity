import os
import sys
import tempfile
import unittest
from collections import OrderedDict

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from scripts.lib.alignment.permutation_pipeline import convert_apply_keys_to_perm_format
from scripts.lib.alignment.vgg16_sinkhorn_alignment import (
    METHOD_PERM_ONLY,
    METHOD_PERM_SCALE,
    VGG16_HIDDEN_LAYER_SPECS,
    VGG16AlignmentParameters,
    apply_alignment_to_state_dict,
    build_hard_alignment_from_indices,
    build_hard_alignment_from_soft,
    run_vgg16_alignment_experiment,
)
from scripts.lib.alignment.vgg16_sinkhorn_evaluation import run_vgg16_alignment_evaluation
from scripts.lib.analysis.alignment import create_vgg16_model
from scripts.lib.transform.random_permutation import VGG16RandomPermutation


def _state_dicts_allclose(state_a, state_b, *, atol=1e-7, rtol=1e-7):
    if set(state_a.keys()) != set(state_b.keys()):
        return False
    for key in state_a:
        if not torch.allclose(state_a[key], state_b[key], atol=atol, rtol=rtol):
            return False
    return True


class VGG16SinkhornScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = create_vgg16_model(num_classes=10, device=cls.device)
        cls.model.eval()
        cls.state = OrderedDict((key, value.detach().cpu().clone()) for key, value in cls.model.state_dict().items())
        cls.model_a_checkpoint = os.path.join(
            PROJECT_ROOT,
            "results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt",
        )
        cls.model_b_checkpoint = os.path.join(
            PROJECT_ROOT,
            "results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt",
        )
        cls.data_path = os.path.join(PROJECT_ROOT, "data")

    def test_identity_biased_logits_with_zero_scales_reproduce_state_after_projection(self):
        aligner = VGG16AlignmentParameters(
            METHOD_PERM_SCALE,
            identity_logit_strength=50.0,
            logit_noise_std=0.0,
        )
        soft_alignment = aligner.build_alignment_matrices(tau=1.0, sinkhorn_iters=20)
        hard_alignment, _ = build_hard_alignment_from_soft(soft_alignment)

        transformed = apply_alignment_to_state_dict(self.state, hard_alignment)
        self.assertTrue(_state_dicts_allclose(self.state, transformed))

    def test_hard_monomial_transform_preserves_logits(self):
        perm_gen = VGG16RandomPermutation()
        permutation_indices = convert_apply_keys_to_perm_format(perm_gen.generate(seed=11))

        generator = torch.Generator()
        generator.manual_seed(7)
        scale_vectors = {
            spec.perm_name: torch.exp(0.1 * torch.randn(spec.size, generator=generator))
            for spec in VGG16_HIDDEN_LAYER_SPECS
        }

        alignment = build_hard_alignment_from_indices(
            permutation_indices,
            scale_vectors,
            device=self.device,
            dtype=torch.float32,
        )
        transformed_state = apply_alignment_to_state_dict(self.state, alignment)

        transformed_model = create_vgg16_model(num_classes=10, device=self.device)
        transformed_model.load_state_dict(transformed_state)
        transformed_model.eval()

        inputs = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            original_logits = self.model(inputs)
            transformed_logits = transformed_model(inputs)

        self.assertTrue(torch.allclose(original_logits, transformed_logits, atol=1e-5, rtol=1e-5))

    def test_perm_only_matches_perm_scale_when_log_scales_are_zero(self):
        perm_only = VGG16AlignmentParameters(METHOD_PERM_ONLY, identity_logit_strength=0.0, logit_noise_std=0.0)
        perm_scale = VGG16AlignmentParameters(METHOD_PERM_SCALE, identity_logit_strength=0.0, logit_noise_std=0.0)

        generator = torch.Generator()
        generator.manual_seed(3)
        for spec in VGG16_HIDDEN_LAYER_SPECS:
            random_logits = torch.randn(spec.size, spec.size, generator=generator)
            perm_only.logits[spec.perm_name].data.copy_(random_logits)
            perm_scale.logits[spec.perm_name].data.copy_(random_logits)
            perm_scale.log_scales[spec.perm_name].data.zero_()

        perm_only_alignment = perm_only.build_alignment_matrices(tau=1.0, sinkhorn_iters=10)
        perm_scale_alignment = perm_scale.build_alignment_matrices(tau=1.0, sinkhorn_iters=10)

        for layer_name in perm_only_alignment.permutations:
            self.assertTrue(
                torch.allclose(
                    perm_only_alignment.permutations[layer_name],
                    perm_scale_alignment.permutations[layer_name],
                    atol=1e-7,
                    rtol=1e-7,
                )
            )
            self.assertTrue(
                torch.allclose(
                    perm_only_alignment.output_monomials[layer_name],
                    perm_scale_alignment.output_monomials[layer_name],
                    atol=1e-7,
                    rtol=1e-7,
                )
            )
            self.assertTrue(
                torch.allclose(
                    perm_only_alignment.input_transports[layer_name],
                    perm_scale_alignment.input_transports[layer_name],
                    atol=1e-7,
                    rtol=1e-7,
                )
            )

    def test_smoke_run_writes_checkpoints_metrics_and_plots(self):
        if not os.path.exists(self.model_a_checkpoint) or not os.path.exists(self.model_b_checkpoint):
            self.skipTest("Expected VGG16 endpoint checkpoints are not available locally.")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = os.path.join(tmpdir, "sinkhorn_scale_smoke")

            train_results = run_vgg16_alignment_experiment(
                model_a_checkpoint=self.model_a_checkpoint,
                model_b_checkpoint=self.model_b_checkpoint,
                output_root=output_root,
                methods=[METHOD_PERM_ONLY, METHOD_PERM_SCALE],
                data_path=self.data_path,
                alpha_grid_train=[0.25, 0.5, 0.75],
                alignment_steps=1,
                alignment_batch_size=8,
                calibration_size=16,
                lr=1e-2,
                tau=1.0,
                sinkhorn_iters=3,
                lambda_scale=1e-5,
                device="cpu",
                num_workers=0,
                seed=0,
                log_interval=1,
            )

            self.assertTrue(os.path.exists(train_results[METHOD_PERM_ONLY].soft_checkpoint_path))
            self.assertTrue(os.path.exists(train_results[METHOD_PERM_ONLY].hard_checkpoint_path))
            self.assertTrue(os.path.exists(train_results[METHOD_PERM_SCALE].artifact_path))

            eval_results = run_vgg16_alignment_evaluation(
                model_a_checkpoint=self.model_a_checkpoint,
                model_b_checkpoint=self.model_b_checkpoint,
                output_root=output_root,
                methods=[METHOD_PERM_ONLY, METHOD_PERM_SCALE],
                data_path=self.data_path,
                num_eval_points=3,
                evaluation_batch_size=8,
                device="cpu",
                num_workers=0,
                max_eval_batches=1,
                plot_filename="smoke_comparison.png",
            )

            expected_files = [
                os.path.join(output_root, "evaluation", "no_alignment", "interpolation.npz"),
                os.path.join(output_root, "evaluation", "sinkhorn_perm_soft", "interpolation.npz"),
                os.path.join(output_root, "evaluation", "sinkhorn_perm_hard", "interpolation.npz"),
                os.path.join(output_root, "evaluation", "sinkhorn_scale_soft", "interpolation.npz"),
                os.path.join(output_root, "evaluation", "sinkhorn_scale_hard", "interpolation.npz"),
                os.path.join(output_root, "evaluation", "comparison.json"),
                os.path.join(output_root, "evaluation", "comparison.csv"),
                os.path.join(output_root, "evaluation", "comparison.md"),
                os.path.join(output_root, "evaluation", "full_summary.json"),
                os.path.join(output_root, "evaluation", "smoke_comparison.png"),
            ]
            for path in expected_files:
                self.assertTrue(os.path.exists(path), msg=f"Missing expected smoke artifact: {path}")

            self.assertEqual(len(eval_results["variant_rows"]), 5)


if __name__ == "__main__":
    unittest.main()
