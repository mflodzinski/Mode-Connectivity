import json
import sys
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

CONFIG_ROOT = PROJECT_ROOT / "configs" / "experiments"
ACTIVE_REFERENCE_ROOTS = (
    PROJECT_ROOT / "experiments",
    PROJECT_ROOT / "src" / "mode_connectivity",
    PROJECT_ROOT / "tools",
    PROJECT_ROOT / "ops" / "slurm",
)
LEGACY_CONFIG_SNIPPETS = (
    "configs/garipov",
    "configs/pytorch_vgg",
    "configs/analysis",
    "configs/config.yaml",
)
RUN_CONFIGS = [
    "curves/runs/curve_seed0_seed1_reg",
    "curves/runs/polygon_seed0_seed1",
    "curves/runs/random_plane_midpoint_seed0_seed1",
    "curves/runs/symmetry_plane_seed0_seed1",
    "lmc/runs/split_30",
    "lmc/runs/resume_shared_checkpoint",
    "sinkhorn/runs/vgg11_cifar_perm_only",
]
XOR_CONFIGS = [
    "xor/runners/permutation_scale.yaml",
    "xor/search/permutation_scale.yaml",
]


class ConfigLayoutTests(unittest.TestCase):
    def test_every_canonical_yaml_loads_with_omegaconf(self):
        for path in CONFIG_ROOT.rglob("*.yaml"):
            with self.subTest(path=path.relative_to(PROJECT_ROOT)):
                cfg = OmegaConf.load(path)
                self.assertIsNotNone(cfg)

    def test_representative_run_configs_compose_with_hydra(self):
        with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
            for config_name in RUN_CONFIGS:
                with self.subTest(config_name=config_name):
                    cfg = compose(config_name=config_name)
                    self.assertIsNotNone(cfg)

    def test_xor_canonical_configs_load(self):
        for relative_path in XOR_CONFIGS:
            with self.subTest(relative_path=relative_path):
                cfg = OmegaConf.load(CONFIG_ROOT / relative_path)
                self.assertIsNotNone(cfg)

    def test_no_legacy_config_tree_references_remain_in_active_code_or_slurm(self):
        for root in ACTIVE_REFERENCE_ROOTS:
            for path in root.rglob("*"):
                if path.suffix not in {".py", ".sh", ".md"}:
                    continue
                text = path.read_text()
                for snippet in LEGACY_CONFIG_SNIPPETS:
                    self.assertNotIn(snippet, text, msg=f"{path} still references {snippet}")

    def test_canonical_tree_has_no_exact_duplicate_yaml_payloads(self):
        seen: dict[str, Path] = {}
        for path in sorted(CONFIG_ROOT.rglob("*.yaml")):
            payload = json.dumps(OmegaConf.to_container(OmegaConf.load(path), resolve=False), sort_keys=True)
            if payload in seen:
                self.fail(f"{path} duplicates {seen[payload]}")
            seen[payload] = path


if __name__ == "__main__":
    unittest.main()
