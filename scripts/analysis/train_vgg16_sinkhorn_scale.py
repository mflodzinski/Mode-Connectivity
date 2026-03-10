"""Train the VGG16 Sinkhorn alignment prototype."""

import os
import sys

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from src.utils import set_global_seed
from scripts.lib.alignment.vgg16_sinkhorn_alignment import run_vgg16_alignment_experiment


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="vgg16_sinkhorn_scale",
)
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))

    run_vgg16_alignment_experiment(
        model_a_checkpoint=to_absolute_path(cfg.model_a_checkpoint),
        model_b_checkpoint=to_absolute_path(cfg.model_b_checkpoint),
        output_root=to_absolute_path(cfg.output_root),
        methods=list(cfg.methods),
        dataset=str(cfg.dataset),
        data_path=to_absolute_path(cfg.data_path),
        alpha_grid_train=list(cfg.alpha_grid_train),
        alignment_steps=int(cfg.alignment_steps),
        alignment_batch_size=int(cfg.alignment_batch_size),
        calibration_size=int(cfg.calibration_size),
        lr=float(cfg.lr),
        tau=float(cfg.tau),
        sinkhorn_iters=int(cfg.sinkhorn_iters),
        lambda_scale=float(cfg.lambda_scale),
        device=str(cfg.device),
        num_workers=int(cfg.num_workers),
        seed=int(cfg.seed),
        log_interval=int(cfg.log_interval),
    )

    output_path = to_absolute_path(cfg.output_root)
    print(f"Alignment artifacts written to: {output_path}")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
