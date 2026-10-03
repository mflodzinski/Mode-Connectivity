import importlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SRC_ROOT = PROJECT_ROOT / "src" / "mode_connectivity"
SLURM_ROOT = PROJECT_ROOT / "ops" / "slurm"
OPTIONAL_DEPS = {
    "torch",
    "torchvision",
    "numpy",
    "scipy",
    "matplotlib",
    "plotly",
    "hydra",
    "omegaconf",
}
RUNNER_MODULES = [
    "experiments.xor.basin_test",
    "experiments.xor.train_linear_barriers",
    "experiments.xor.permutation_scale",
    "experiments.xor.curve_fitting",
    "experiments.xor.joint_scale_polygonal",
    "experiments.training_stage.run",
    "experiments.fashion_mnist.run",
    "experiments.dense_linear_stage.run",
    "tools.plotting.plot_larger_network_barriers",
]


class RepoStructureTests(unittest.TestCase):
    def test_legacy_runner_packages_removed_from_src(self):
        self.assertFalse((SRC_ROOT / "experiments").exists())
        self.assertFalse((SRC_ROOT / "plotting").exists())
        self.assertFalse((SRC_ROOT / "verification").exists())

    def test_src_tree_does_not_import_legacy_runner_packages(self):
        banned_snippets = (
            "mode_connectivity.experiments.",
            "mode_connectivity.plotting.",
            "mode_connectivity.verification.",
        )
        for path in SRC_ROOT.rglob("*.py"):
            text = path.read_text()
            for snippet in banned_snippets:
                self.assertNotIn(snippet, text, msg=f"{path} still references {snippet}")

    def test_runner_modules_import(self):
        for module_name in RUNNER_MODULES:
            with self.subTest(module=module_name):
                try:
                    importlib.import_module(module_name)
                except ModuleNotFoundError as exc:
                    if exc.name in OPTIONAL_DEPS:
                        self.skipTest(f"Optional dependency missing for import smoke test: {exc.name}")
                    raise

    def test_slurm_tree_uses_current_layout_only(self):
        self.assertTrue((SLURM_ROOT / "common.sh").exists())
        self.assertTrue((SLURM_ROOT / "xor").exists())
        self.assertTrue((SLURM_ROOT / "training_stage").exists())
        self.assertTrue((SLURM_ROOT / "fashion_mnist").exists())
        self.assertTrue((SLURM_ROOT / "dense_linear_stage").exists())
        self.assertFalse((SLURM_ROOT / "analysis").exists())
        self.assertFalse((SLURM_ROOT / "advanced_geometry").exists())
        self.assertFalse((SLURM_ROOT / "eval").exists())
        self.assertFalse((SLURM_ROOT / "lmc_connected").exists())
        for path in SLURM_ROOT.rglob("*.sh"):
            text = path.read_text()
            self.assertNotIn("scripts/slurm/", text, msg=f"{path} still references removed scripts/slurm paths")


if __name__ == "__main__":
    unittest.main()
