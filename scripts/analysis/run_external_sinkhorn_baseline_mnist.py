"""Run the vendored external/sinkhorn-rebasin baseline on VGG16/MNIST endpoints."""

import os
import sys

import hydra
from omegaconf import DictConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from scripts.analysis.run_external_sinkhorn_baseline import run_external_sinkhorn_baseline


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_rebasin_vgg16_mnist",
)
def main(cfg: DictConfig) -> None:
    run_external_sinkhorn_baseline(cfg)


if __name__ == "__main__":
    main()
