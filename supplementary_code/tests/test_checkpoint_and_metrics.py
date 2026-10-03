import os
import sys
import unittest
from collections import OrderedDict

try:
    import torch
    _TORCH_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # pragma: no cover - environment-dependent
    torch = None
    _TORCH_IMPORT_ERROR = exc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

if torch is not None:
    from mode_connectivity.core.checkpoint import extract_state_dict, normalize_state_dict_keys
    from mode_connectivity.evaluation.metrics import l2_distance, state_distance_summary


class CheckpointAndMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _TORCH_IMPORT_ERROR is not None:  # pragma: no cover - environment-dependent
            raise unittest.SkipTest(f"torch is not installed: {_TORCH_IMPORT_ERROR}")

    def test_extract_state_dict_accepts_raw_state_dict(self):
        state = OrderedDict({"weight": torch.tensor([1.0, 2.0])})
        extracted = extract_state_dict(state)
        self.assertEqual(list(extracted.keys()), ["weight"])
        self.assertTrue(torch.equal(extracted["weight"], state["weight"]))

    def test_extract_state_dict_accepts_model_state_payload(self):
        payload = {"model_state": OrderedDict({"weight": torch.tensor([3.0])})}
        extracted = extract_state_dict(payload)
        self.assertTrue(torch.equal(extracted["weight"], torch.tensor([3.0])))

    def test_extract_state_dict_accepts_state_dict_payload(self):
        payload = {"state_dict": OrderedDict({"bias": torch.tensor([4.0])})}
        extracted = extract_state_dict(payload)
        self.assertTrue(torch.equal(extracted["bias"], torch.tensor([4.0])))

    def test_normalize_state_dict_keys_handles_module_prefixes(self):
        state = OrderedDict(
            {
                "module.classifier.weight": torch.tensor([1.0]),
                "features.module.0.weight": torch.tensor([2.0]),
            }
        )
        normalized = normalize_state_dict_keys(state)
        self.assertIn("classifier.weight", normalized)
        self.assertIn("features.0.weight", normalized)
        self.assertNotIn("module.classifier.weight", normalized)
        self.assertNotIn("features.module.0.weight", normalized)

    def test_l2_distance_zero_for_identical_states(self):
        state = OrderedDict({"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([3.0])})
        metrics = l2_distance(state, state)
        self.assertEqual(metrics["total_l2"], 0.0)
        self.assertEqual(metrics["normalized_total_l2"], 0.0)
        self.assertEqual(metrics["total_params"], 3)

    def test_l2_distance_reports_per_layer_and_summary_schema(self):
        state_a = OrderedDict({"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([1.0])})
        state_b = OrderedDict({"weight": torch.tensor([1.0, 5.0]), "bias": torch.tensor([5.0])})
        metrics = l2_distance(state_a, state_b, compute_per_layer=True)
        summary = state_distance_summary(state_a, state_b)
        self.assertIn("layer_distances", metrics)
        self.assertIn("weight", metrics["layer_distances"])
        self.assertAlmostEqual(metrics["total_l2"], 5.0)
        self.assertAlmostEqual(summary["l2_distance"], 5.0)
        self.assertAlmostEqual(summary["rms_difference"], 5.0 / (3.0 ** 0.5))

    def test_l2_distance_rejects_shape_mismatches(self):
        state_a = OrderedDict({"weight": torch.tensor([1.0, 2.0])})
        state_b = OrderedDict({"weight": torch.tensor([[1.0, 2.0]])})
        with self.assertRaises(ValueError):
            l2_distance(state_a, state_b)


if __name__ == "__main__":
    unittest.main()
