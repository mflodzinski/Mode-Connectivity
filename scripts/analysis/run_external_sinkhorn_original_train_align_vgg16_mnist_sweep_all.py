"""Train original sinkhorn VGG16 MNIST endpoints once, then sweep alignment params."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_baseline_sweep import (
    build_output_tag,
    enumerate_sweep_combinations,
)
from scripts.analysis.run_external_sinkhorn_original_train_align_vgg16_mnist import (
    run_alignment_from_models,
    train_endpoints,
)
from scripts.lib.alignment.permutation_pipeline import write_summary_files
from scripts.lib.core.output import ensure_dir, load_json, save_json


def load_run_comparison(output_root: str) -> Dict[str, Dict[str, Any]]:
    comparison_path = Path(output_root) / "evaluation" / "comparison.json"
    rows = load_json(comparison_path)
    return {row["variant_key"]: row for row in rows}


def build_sweep_comparison_row(run_record: Dict[str, Any]) -> Dict[str, Any]:
    comparison = load_run_comparison(run_record["output_root"])
    hard = comparison["original_sinkhorn_hard"]
    soft = comparison["original_sinkhorn_soft"]
    no_alignment = comparison["no_alignment"]
    combo = run_record["combo"]
    return {
        "task_id": run_record["task_id"],
        "output_tag": run_record["output_tag"],
        "output_root": run_record["output_root"],
        "alignment_epochs": combo["alignment_epochs"],
        "loss_name": combo["loss_name"],
        "tau": combo["tau"],
        "lr": combo["lr"],
        "sinkhorn_l": combo["sinkhorn_l"],
        "hard_endpoint_b_test_acc": hard["endpoint_b_test_acc"],
        "hard_mean_test_interp_loss": hard["mean_test_interp_loss"],
        "hard_test_loss_barrier_avg": hard["test_loss_barrier_avg"],
        "hard_test_loss_barrier_max_endpoint": hard["test_loss_barrier_max_endpoint"],
        "hard_min_test_acc": hard["min_test_acc"],
        "soft_endpoint_b_test_acc": soft["endpoint_b_test_acc"],
        "soft_mean_test_interp_loss": soft["mean_test_interp_loss"],
        "soft_test_loss_barrier_avg": soft["test_loss_barrier_avg"],
        "soft_test_loss_barrier_max_endpoint": soft["test_loss_barrier_max_endpoint"],
        "soft_min_test_acc": soft["min_test_acc"],
        "no_align_mean_test_interp_loss": no_alignment["mean_test_interp_loss"],
        "no_align_test_loss_barrier_max_endpoint": no_alignment["test_loss_barrier_max_endpoint"],
        "no_align_min_test_acc": no_alignment["min_test_acc"],
        "delta_hard_mean_test_interp_loss_vs_no_align": hard["mean_test_interp_loss"] - no_alignment["mean_test_interp_loss"],
        "delta_hard_test_loss_barrier_max_endpoint_vs_no_align": (
            hard["test_loss_barrier_max_endpoint"] - no_alignment["test_loss_barrier_max_endpoint"]
        ),
        "delta_hard_min_test_acc_vs_no_align": hard["min_test_acc"] - no_alignment["min_test_acc"],
    }


def print_top_runs(rows: list[Dict[str, Any]], *, top_k: int = 5) -> None:
    if not rows:
        print("No completed runs available for sweep comparison.")
        return

    top_barrier = sorted(rows, key=lambda row: row["hard_test_loss_barrier_max_endpoint"])[:top_k]
    top_mean = sorted(rows, key=lambda row: row["hard_mean_test_interp_loss"])[:top_k]

    print("")
    print("=" * 80)
    print("TOP RUNS BY HARD TEST BARRIER")
    print("=" * 80)
    for row in top_barrier:
        print(
            f"task={row['task_id']:02d} "
            f"epochs={row['alignment_epochs']} loss={row['loss_name']} tau={row['tau']} "
            f"lr={row['lr']} l={row['sinkhorn_l']} "
            f"hard_barrier={row['hard_test_loss_barrier_max_endpoint']:.4f} "
            f"hard_mean={row['hard_mean_test_interp_loss']:.4f} "
            f"hard_min_acc={row['hard_min_test_acc']:.2f}"
        )

    print("")
    print("=" * 80)
    print("TOP RUNS BY HARD MEAN TEST LOSS")
    print("=" * 80)
    for row in top_mean:
        print(
            f"task={row['task_id']:02d} "
            f"epochs={row['alignment_epochs']} loss={row['loss_name']} tau={row['tau']} "
            f"lr={row['lr']} l={row['sinkhorn_l']} "
            f"hard_mean={row['hard_mean_test_interp_loss']:.4f} "
            f"hard_barrier={row['hard_test_loss_barrier_max_endpoint']:.4f} "
            f"hard_min_acc={row['hard_min_test_acc']:.2f}"
        )


def run_train_then_sweep(cfg: DictConfig) -> None:
    combos = enumerate_sweep_combinations(cfg.sweep)
    total_runs = len(combos)

    start_index = int(cfg.get("start_index", 0))
    end_index_cfg = cfg.get("end_index", None)
    end_index = total_runs - 1 if end_index_cfg is None else int(end_index_cfg)
    continue_on_error = bool(cfg.get("continue_on_error", False))

    if start_index < 0 or start_index >= total_runs:
        raise ValueError(f"start_index={start_index} is out of range for {total_runs} runs.")
    if end_index < start_index or end_index >= total_runs:
        raise ValueError(
            f"end_index={end_index} is invalid for start_index={start_index} and total_runs={total_runs}."
        )

    base_output_root = ensure_dir(Path(to_absolute_path(str(cfg.base_output_root))))
    train_cfg = OmegaConf.create(
        {
            "experiment_name": f"{cfg.experiment_name}_endpoint_training",
            "output_root": str(base_output_root),
            "data_path": to_absolute_path(str(cfg.data_path)),
            "use_small_dataset": bool(cfg.use_small_dataset),
            "validation_size": int(cfg.validation_size),
            "image_size": int(cfg.image_size),
            "num_workers": int(cfg.num_workers),
            "split_seed": int(cfg.split_seed),
            "model_a_seed": int(cfg.model_a_seed),
            "model_b_seed": int(cfg.model_b_seed),
            "train_batch_size": int(cfg.train_batch_size),
            "eval_batch_size": int(cfg.eval_batch_size),
            "train_epochs": int(cfg.train_epochs),
            "train_lr": float(cfg.train_lr),
            "device": str(cfg.device),
        }
    )

    print("=" * 80)
    print("ORIGINAL SINKHORN VGG16 MNIST TRAIN + ALIGN SWEEP")
    print("=" * 80)
    print(f"total configured runs: {total_runs}")
    print(f"selected range: {start_index}..{end_index}")
    print(f"base_output_root: {base_output_root}")
    print("")

    trained = train_endpoints(train_cfg)
    save_json(
        {
            "model_a_path": str(trained["model_a_path"]),
            "model_b_path": str(trained["model_b_path"]),
            "model_a_metrics": trained["model_a_metrics"],
            "model_b_metrics": trained["model_b_metrics"],
            "train_config": OmegaConf.to_container(train_cfg, resolve=True),
        },
        base_output_root / "trained_endpoints.json",
        indent=2,
    )

    sweep_summary: Dict[str, Any] = {
        "experiment_name": str(cfg.experiment_name),
        "base_output_root": str(base_output_root),
        "trained_endpoints": {
            "model_a_path": str(trained["model_a_path"]),
            "model_b_path": str(trained["model_b_path"]),
            "model_a_metrics": trained["model_a_metrics"],
            "model_b_metrics": trained["model_b_metrics"],
        },
        "total_configured_runs": total_runs,
        "start_index": start_index,
        "end_index": end_index,
        "continue_on_error": continue_on_error,
        "runs": [],
    }

    for task_id in range(start_index, end_index + 1):
        combo = combos[task_id]
        output_tag = build_output_tag(
            {
                "alignment_steps": combo["alignment_epochs"],
                "tau": combo["tau"],
                "lr": combo["lr"],
                "sinkhorn_l": combo["sinkhorn_l"],
            }
        ) + f"_loss{combo['loss_name']}"
        output_root = str(base_output_root / "sweep_runs" / output_tag)
        run_cfg = OmegaConf.create(
            {
                "experiment_name": f"{cfg.experiment_name}_{output_tag}",
                "output_root": output_root,
                "image_size": int(cfg.image_size),
                "alignment_seed": int(cfg.alignment_seed),
                "alignment_epochs": int(combo["alignment_epochs"]),
                "alignment_lr": float(combo["lr"]),
                "loss_name": str(combo["loss_name"]),
                "tau": float(combo["tau"]),
                "sinkhorn_iters": int(cfg.sinkhorn_iters),
                "sinkhorn_l": float(combo["sinkhorn_l"]),
                "identity_init": bool(cfg.identity_init),
                "log_interval": int(cfg.log_interval),
                "num_eval_points": int(cfg.num_eval_points),
                "plot_filename": str(cfg.plot_filename),
            }
        )

        print("-" * 80)
        print(f"run {task_id + 1}/{total_runs}")
        print(f"task_id: {task_id}")
        print(f"combo: {combo}")
        print(f"output_root: {output_root}")
        print("")

        run_record: Dict[str, Any] = {
            "task_id": task_id,
            "output_tag": output_tag,
            "output_root": output_root,
            "combo": combo,
        }

        try:
            metadata = run_alignment_from_models(
                run_cfg,
                model_a=deepcopy(trained["model_a"]),
                model_b=deepcopy(trained["model_b"]),
                model_a_path=trained["model_a_path"],
                model_b_path=trained["model_b_path"],
                model_a_metrics=trained["model_a_metrics"],
                model_b_metrics=trained["model_b_metrics"],
                train_loader=trained["train_loader"],
                val_loader=trained["val_loader"],
                test_loader=trained["test_loader"],
                eval_loss_acc=trained["eval_loss_acc"],
                lerp=trained["lerp"],
                device=trained["device"],
                include_no_alignment=True,
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
    if comparison_rows:
        comparison_rows = sorted(comparison_rows, key=lambda row: row["hard_test_loss_barrier_max_endpoint"])
        write_summary_files(str(base_output_root / "sweep_comparison"), comparison_rows)
        save_json(comparison_rows, base_output_root / "sweep_comparison.json", indent=2)
        print_top_runs(comparison_rows, top_k=min(5, len(comparison_rows)))

    print("")
    print("=" * 80)
    print("ORIGINAL SINKHORN VGG16 MNIST TRAIN + ALIGN SWEEP COMPLETE")
    print("=" * 80)
    print(f"Endpoint summary: {base_output_root / 'trained_endpoints.json'}")
    print(f"Sweep summary: {base_output_root / 'sweep_summary.json'}")
    if comparison_rows:
        print(f"Comparison table: {base_output_root / 'sweep_comparison.json'}")


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_train_align_vgg16_mnist_sweep",
)
def main(cfg: DictConfig) -> None:
    run_train_then_sweep(cfg)


if __name__ == "__main__":
    main()
