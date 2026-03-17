"""Run the original external/sinkhorn-rebasin LMC loop on VGG16/CIFAR10 endpoints."""

from __future__ import annotations

import os
import sys

import hydra
from omegaconf import DictConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from scripts.analysis.run_external_sinkhorn_original_lmc_vgg16_mnist import (
    run_original_sinkhorn_lmc_vgg16_mnist,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_lmc_vgg16_cifar10",
)
def main(cfg: DictConfig) -> None:
    run_original_sinkhorn_lmc_vgg16_mnist(cfg)


if __name__ == "__main__":
    main()
