"""Run the full configured original external Sinkhorn LMC CIFAR10 sweep."""

from __future__ import annotations

import os
import sys

import hydra
from omegaconf import DictConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from scripts.analysis.run_external_sinkhorn_original_lmc_vgg16_mnist_sweep_all import (
    run_original_sinkhorn_lmc_sweep_all,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_lmc_vgg16_cifar10_sweep",
)
def main(cfg: DictConfig) -> None:
    run_original_sinkhorn_lmc_sweep_all(cfg)


if __name__ == "__main__":
    main()
