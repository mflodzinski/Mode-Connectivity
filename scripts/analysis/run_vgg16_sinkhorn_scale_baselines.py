"""Run VGG16/CIFAR10 baseline comparisons for the Sinkhorn+scale prototype.

This is the single entrypoint intended for cluster jobs. It trains the learned
alignment baselines and then evaluates the comparison set that includes:
1. No alignment
2. Sinkhorn permutation-only re-basin
3. Sinkhorn + diagonal scaling
"""

import os
import sys
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "scripts"))

from src.utils import set_global_seed
from scripts.lib.alignment.vgg16_sinkhorn_alignment import run_vgg16_alignment_experiment
from scripts.lib.alignment.vgg16_sinkhorn_evaluation import run_vgg16_alignment_evaluation
from scripts.lib.core.output import save_json


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="vgg16_sinkhorn_scale",
)
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))

    output_root = to_absolute_path(cfg.output_root)
    model_a_checkpoint = to_absolute_path(cfg.model_a_checkpoint)
    model_b_checkpoint = to_absolute_path(cfg.model_b_checkpoint)
    data_path = to_absolute_path(cfg.data_path)
    methods = list(cfg.methods)

    print("=" * 80)
    print("VGG16/CIFAR10 SINKHORN BASELINE PIPELINE")
    print("=" * 80)
    print(f"experiment_name: {cfg.experiment_name}")
    print(f"model_a_checkpoint: {model_a_checkpoint}")
    print(f"model_b_checkpoint: {model_b_checkpoint}")
    print(f"output_root: {output_root}")
    print(f"methods: {methods}")
    print(f"device: {cfg.device}")
    print("")

    print("=" * 80)
    print("STAGE 1: TRAIN ALIGNMENT BASELINES")
    print("=" * 80)
    train_results = run_vgg16_alignment_experiment(
        model_a_checkpoint=model_a_checkpoint,
        model_b_checkpoint=model_b_checkpoint,
        output_root=output_root,
        methods=methods,
        data_path=data_path,
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

    print("")
    print("=" * 80)
    print("STAGE 2: EVALUATE BASELINE COMPARISONS")
    print("=" * 80)
    eval_results = run_vgg16_alignment_evaluation(
        model_a_checkpoint=model_a_checkpoint,
        model_b_checkpoint=model_b_checkpoint,
        output_root=output_root,
        methods=methods,
        data_path=data_path,
        num_eval_points=int(cfg.num_eval_points),
        evaluation_batch_size=int(cfg.evaluation_batch_size),
        device=str(cfg.device),
        num_workers=int(cfg.num_workers),
        max_eval_batches=None if cfg.max_eval_batches is None else int(cfg.max_eval_batches),
        plot_filename=str(cfg.plot_filename),
    )

    pipeline_summary = {
        "experiment_name": str(cfg.experiment_name),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "training_outputs": {
            method: {
                "method_dir": result.method_dir,
                "soft_checkpoint_path": result.soft_checkpoint_path,
                "hard_checkpoint_path": result.hard_checkpoint_path,
                "artifact_path": result.artifact_path,
                "history_path": result.history_path,
                "metadata_path": result.metadata_path,
            }
            for method, result in train_results.items()
        },
        "evaluation_outputs": eval_results,
    }
    summary_path = Path(output_root) / "pipeline_summary.json"
    save_json(pipeline_summary, summary_path)

    print("")
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"Training outputs: {output_root}")
    print(f"Evaluation directory: {eval_results['evaluation_dir']}")
    print(f"Comparison plot: {eval_results['plot_path']}")
    print(f"Pipeline summary: {summary_path}")


if __name__ == "__main__":
    main()
