"""Sweep original sinkhorn re-basin alignment params for saved non-BN VGG MNIST endpoints."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
import torchvision
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_baseline_sweep import enumerate_sweep_combinations
from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import import_original_mnist_components
from scripts.analysis.run_external_sinkhorn_original_vgg_cifar10_align_sweep_all import (
    build_output_tag,
    build_sweep_comparison_row,
    print_top_runs,
    run_one_alignment,
)
from scripts.lib.alignment.permutation_pipeline import resolve_device, write_summary_files
from scripts.lib.core.output import ensure_dir, save_json
from src.utils import set_global_seed


def build_mnist_loaders(cfg: DictConfig, dnn_data):
    transform_train = dnn_data.Transforms.MNIST.VGG.train
    transform_test = dnn_data.Transforms.MNIST.VGG.test
    mnist_root = Path(to_absolute_path(str(cfg.data_path))) / "mnist"

    dataset_train_source = torchvision.datasets.MNIST(
        root=mnist_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    dataset_val_source = torchvision.datasets.MNIST(
        root=mnist_root,
        train=True,
        download=True,
        transform=transform_test,
    )
    dataset_test = torchvision.datasets.MNIST(
        root=mnist_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    full_train_size = len(dataset_train_source)
    val_size = int(full_train_size * float(cfg.val_fraction))
    train_size = full_train_size - val_size
    generator = torch.Generator().manual_seed(int(cfg.split_seed))
    shuffled_indices = torch.randperm(full_train_size, generator=generator).tolist()
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:]

    dataset_train = torch.utils.data.Subset(dataset_train_source, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val_source, val_indices)

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers))
    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers))
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers))
    return train_loader, val_loader, test_loader


def run_alignment_sweep_all(cfg: DictConfig) -> None:
    combos = enumerate_sweep_combinations(cfg.sweep)
    total_runs = len(combos)
    start_index = int(cfg.get("start_index", 0))
    end_index_cfg = cfg.get("end_index", None)
    end_index = total_runs - 1 if end_index_cfg is None else int(end_index_cfg)
    continue_on_error = bool(cfg.get("continue_on_error", False))

    if start_index < 0 or start_index >= total_runs:
        raise ValueError(f"start_index={start_index} is out of range for {total_runs} runs.")
    if end_index < start_index or end_index >= total_runs:
        raise ValueError(f"end_index={end_index} is invalid for start_index={start_index} and total_runs={total_runs}.")

    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    base_output_root = ensure_dir(Path(to_absolute_path(str(cfg.base_output_root))))

    VGGClass, RebasinNet, matching, dnn_data, RndLoss, eval_loss_acc, lerp = import_original_mnist_components()
    from rebasin.loss import DistL1Loss, DistL2Loss, MidLoss

    train_loader, val_loader, test_loader = build_mnist_loaders(cfg, dnn_data)

    sweep_summary: Dict[str, Any] = {
        "experiment_name": str(cfg.experiment_name),
        "base_output_root": str(base_output_root),
        "model_a_checkpoint": str(to_absolute_path(str(cfg.model_a_checkpoint))),
        "model_b_checkpoint": str(to_absolute_path(str(cfg.model_b_checkpoint))),
        "total_configured_runs": total_runs,
        "start_index": start_index,
        "end_index": end_index,
        "continue_on_error": continue_on_error,
        "runs": [],
    }

    print("=" * 80)
    print(f"ORIGINAL SINKHORN {cfg.vgg_name} MNIST ALIGNMENT SWEEP")
    print("=" * 80)
    print(f"total configured runs: {total_runs}")
    print(f"selected range: {start_index}..{end_index}")
    print(f"base_output_root: {base_output_root}")
    print(f"model_a_checkpoint: {to_absolute_path(str(cfg.model_a_checkpoint))}")
    print(f"model_b_checkpoint: {to_absolute_path(str(cfg.model_b_checkpoint))}")
    print("")

    for task_id in range(start_index, end_index + 1):
        combo = combos[task_id]
        output_tag = build_output_tag(combo, cfg)
        output_root = str(base_output_root / output_tag)
        run_cfg = OmegaConf.create(
            {
                "experiment_name": f"{cfg.experiment_name}_{output_tag}",
                "dataset_name": "MNIST",
                "vgg_name": str(cfg.vgg_name),
                "model_a_checkpoint": to_absolute_path(str(cfg.model_a_checkpoint)),
                "model_b_checkpoint": to_absolute_path(str(cfg.model_b_checkpoint)),
                "output_root": output_root,
                "data_path": to_absolute_path(str(cfg.data_path)),
                "image_size": int(cfg.image_size),
                "val_fraction": float(cfg.val_fraction),
                "split_seed": int(cfg.split_seed),
                "alignment_iterations": int(combo["alignment_iterations"]),
                "alignment_lr": float(combo["lr"]),
                "loss_name": str(combo["loss_name"]),
                "tau": float(combo["tau"]),
                "sinkhorn_iters": int(cfg.sinkhorn_iters),
                "sinkhorn_l": float(combo["sinkhorn_l"]),
                "identity_init": bool(cfg.identity_init),
                "scale_invariant": bool(cfg.get("scale_invariant", False)),
                "lambda_scale": float(combo["lambda_scale"]) if "lambda_scale" in combo else float(cfg.get("lambda_scale", 1e-4)),
                "best_eval_interval": int(cfg.best_eval_interval),
                "validation_alpha_grid": [float(alpha) for alpha in cfg.validation_alpha_grid],
                "early_stopping_patience": int(cfg.get("early_stopping_patience", 0)),
                "early_stopping_min_delta": float(cfg.get("early_stopping_min_delta", 0.0)),
                "starting_alignment_artifact": cfg.get("starting_alignment_artifact", None),
                "starting_permutation_kind": str(cfg.get("starting_permutation_kind", "hard")),
                "finetune_mode": str(cfg.get("finetune_mode", "joint")),
                "num_eval_points": int(cfg.num_eval_points),
                "log_interval": int(cfg.log_interval),
            }
        )
        print("-" * 80)
        print(f"run {task_id + 1}/{total_runs}")
        print(f"task_id: {task_id}")
        print(f"combo: {combo}")
        print(f"output_root: {output_root}")
        print("")

        run_record: Dict[str, Any] = {"task_id": task_id, "output_tag": output_tag, "output_root": output_root, "combo": combo}
        try:
            metadata = run_one_alignment(
                run_cfg,
                VGGClass=VGGClass,
                RebasinNet=RebasinNet,
                matching=matching,
                MidLoss=MidLoss,
                RndLoss=RndLoss,
                DistL1Loss=DistL1Loss,
                DistL2Loss=DistL2Loss,
                eval_loss_acc=eval_loss_acc,
                lerp=lerp,
                device=device,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
            )
            run_record["status"] = "completed"
            run_record["run_metadata"] = metadata
        except Exception as exc:
            run_record["status"] = "failed"
            run_record["error"] = repr(exc)
            sweep_summary["runs"].append(run_record)
            save_json(sweep_summary, base_output_root / "sweep_summary.json", indent=2)
            if not continue_on_error:
                raise
            print(f"run failed and continue_on_error=true: {exc}")
            continue

        sweep_summary["runs"].append(run_record)
        save_json(sweep_summary, base_output_root / "sweep_summary.json", indent=2)

    completed_runs = [run for run in sweep_summary["runs"] if run.get("status") == "completed"]
    comparison_rows = [build_sweep_comparison_row(run) for run in completed_runs]
    save_json(comparison_rows, base_output_root / "sweep_comparison.json", indent=2)
    write_summary_files(str(base_output_root / "sweep_comparison"), comparison_rows)
    print_top_runs(comparison_rows)


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_vgg_mnist_align_sweep",
)
def main(cfg: DictConfig) -> None:
    run_alignment_sweep_all(cfg)


if __name__ == "__main__":
    main()
