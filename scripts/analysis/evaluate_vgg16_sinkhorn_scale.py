"""Evaluate the VGG16 Sinkhorn alignment prototype."""

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
from scripts.lib.alignment.vgg16_sinkhorn_evaluation import run_vgg16_alignment_evaluation


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="vgg16_sinkhorn_scale",
)
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))

    result = run_vgg16_alignment_evaluation(
        model_a_checkpoint=to_absolute_path(cfg.model_a_checkpoint),
        model_b_checkpoint=to_absolute_path(cfg.model_b_checkpoint),
        output_root=to_absolute_path(cfg.output_root),
        methods=list(cfg.methods),
        dataset=str(cfg.dataset),
        data_path=to_absolute_path(cfg.data_path),
        num_eval_points=int(cfg.num_eval_points),
        evaluation_batch_size=int(cfg.evaluation_batch_size),
        device=str(cfg.device),
        num_workers=int(cfg.num_workers),
        max_eval_batches=None if cfg.max_eval_batches is None else int(cfg.max_eval_batches),
        plot_filename=str(cfg.plot_filename),
    )

    print(f"Evaluation artifacts written to: {result['evaluation_dir']}")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
