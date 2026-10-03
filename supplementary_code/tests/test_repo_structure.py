import importlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from omegaconf import OmegaConf

from mode_connectivity.curves.runners import build_curve_training_command


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
    "experiments.curves.garipov_curve",
    "experiments.curves.garipov_endpoints",
    "experiments.lmc.pytorch_vgg16_lmc_connected_pair",
    "experiments.lmc.pytorch_vgg16_lmc_connected_pair_from_scratch",
    "experiments.sinkhorn.vgg_cifar_alignment_sweep",
    "experiments.xor.permutation_scale",
    "tools.plotting.plot_pytorch_vgg_split_suite",
    "tools.verification.network_transform",
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

    def test_curve_runner_command_builder_centralizes_shared_flags(self):
        cfg = OmegaConf.create(
            {
                "dataset": "CIFAR10",
                "data_path": "./data",
                "transform": "VGG",
                "model": "VGG16",
                "epochs": 200,
                "lr": 0.05,
                "wd": 0.0005,
                "curve": "Bezier",
                "num_bends": 3,
                "seed": 7,
                "save_intermediate": True,
                "save_freq": 50,
                "use_test": True,
                "no_train_aug": True,
                "train_half_only": False,
                "batch_size": 128,
                "momentum": 0.9,
                "num_workers": 4,
            }
        )
        cmd = build_curve_training_command(
            cfg=cfg,
            output_dir="results/demo",
            endpoint0="/tmp/a.pt",
            endpoint1="/tmp/b.pt",
            curve_type="PolyChain",
            num_bends=3,
            include_training_hparams=True,
            extra_flags=["--project_symmetry_plane"],
        )
        self.assertIn("--curve", cmd)
        self.assertIn("PolyChain", cmd)
        self.assertIn("--init_start", cmd)
        self.assertIn("/tmp/a.pt", cmd)
        self.assertIn("--fix_start", cmd)
        self.assertIn("--seed", cmd)
        self.assertIn("7", cmd)
        self.assertIn("--project_symmetry_plane", cmd)

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
        self.assertTrue((SLURM_ROOT / "curves").exists())
        self.assertTrue((SLURM_ROOT / "endpoints").exists())
        self.assertTrue((SLURM_ROOT / "lmc").exists())
        self.assertTrue((SLURM_ROOT / "sinkhorn").exists())
        self.assertTrue((SLURM_ROOT / "xor").exists())
        self.assertTrue((SLURM_ROOT / "verification").exists())
        self.assertFalse((SLURM_ROOT / "analysis").exists())
        self.assertFalse((SLURM_ROOT / "advanced_geometry").exists())
        self.assertFalse((SLURM_ROOT / "eval").exists())
        self.assertFalse((SLURM_ROOT / "lmc_connected").exists())
        for path in SLURM_ROOT.rglob("*.sh"):
            text = path.read_text()
            self.assertNotIn("scripts/slurm/", text, msg=f"{path} still references removed scripts/slurm paths")


if __name__ == "__main__":
    unittest.main()
