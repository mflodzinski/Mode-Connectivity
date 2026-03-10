"""Run the full configured external Sinkhorn MNIST sweep sequentially."""

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_baseline_sweep_all import run_external_sinkhorn_sweep_all


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_rebasin_vgg16_mnist_sweep",
)
def main(cfg: DictConfig) -> None:
    run_external_sinkhorn_sweep_all(cfg)


if __name__ == "__main__":
    main()
