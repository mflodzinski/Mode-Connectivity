"""Run the full configured external Sinkhorn sweep sequentially.

This is the non-array launcher: it reads the sweep grid from the Hydra config
and invokes ``run_external_sinkhorn_baseline`` once per configuration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_baseline import run_external_sinkhorn_baseline
from scripts.analysis.run_external_sinkhorn_baseline_sweep import (
    build_output_tag,
    enumerate_sweep_combinations,
)
from scripts.lib.alignment.permutation_pipeline import write_summary_files
from scripts.lib.core.output import ensure_dir, load_json, save_json


def load_run_comparison(output_root: str) -> Dict[str, Dict[str, Any]]:
    """Load one run's variant comparison rows keyed by variant_key."""

    comparison_path = Path(output_root) / "evaluation" / "comparison.json"
    rows = load_json(comparison_path)
    return {row["variant_key"]: row for row in rows}


def build_sweep_comparison_row(run_record: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one completed run into a sweep-level comparison row."""

    comparison = load_run_comparison(run_record["output_root"])
    no_alignment = comparison["no_alignment"]
    hard = comparison["external_sinkhorn_hard"]
    soft = comparison["external_sinkhorn_soft"]
    combo = run_record["combo"]

    return {
        "task_id": run_record["task_id"],
        "output_tag": run_record["output_tag"],
        "output_root": run_record["output_root"],
        "alignment_steps": combo["alignment_steps"],
        "tau": combo["tau"],
        "lr": combo["lr"],
        "sinkhorn_l": combo["sinkhorn_l"],
        "hard_endpoint_b_test_acc": hard["endpoint_b_test_acc"],
        "hard_mean_test_interp_loss": hard["mean_test_interp_loss"],
        "hard_test_loss_barrier_avg": hard["test_loss_barrier_avg"],
        "hard_test_loss_barrier_max_endpoint": hard["test_loss_barrier_max_endpoint"],
        "hard_min_test_acc": hard["min_test_acc"],
        "hard_test_acc_drop_from_endpoint_min": hard["test_acc_drop_from_endpoint_min"],
        "soft_endpoint_b_test_acc": soft["endpoint_b_test_acc"],
        "soft_mean_test_interp_loss": soft["mean_test_interp_loss"],
        "soft_test_loss_barrier_avg": soft["test_loss_barrier_avg"],
        "soft_test_loss_barrier_max_endpoint": soft["test_loss_barrier_max_endpoint"],
        "soft_min_test_acc": soft["min_test_acc"],
        "soft_test_acc_drop_from_endpoint_min": soft["test_acc_drop_from_endpoint_min"],
        "no_align_mean_test_interp_loss": no_alignment["mean_test_interp_loss"],
        "no_align_test_loss_barrier_avg": no_alignment["test_loss_barrier_avg"],
        "no_align_test_loss_barrier_max_endpoint": no_alignment["test_loss_barrier_max_endpoint"],
        "no_align_min_test_acc": no_alignment["min_test_acc"],
        "delta_hard_mean_test_interp_loss_vs_no_align": (
            hard["mean_test_interp_loss"] - no_alignment["mean_test_interp_loss"]
        ),
        "delta_hard_test_loss_barrier_avg_vs_no_align": (
            hard["test_loss_barrier_avg"] - no_alignment["test_loss_barrier_avg"]
        ),
        "delta_hard_test_loss_barrier_max_endpoint_vs_no_align": (
            hard["test_loss_barrier_max_endpoint"] - no_alignment["test_loss_barrier_max_endpoint"]
        ),
        "delta_hard_min_test_acc_vs_no_align": hard["min_test_acc"] - no_alignment["min_test_acc"],
        "delta_soft_mean_test_interp_loss_vs_no_align": (
            soft["mean_test_interp_loss"] - no_alignment["mean_test_interp_loss"]
        ),
        "delta_soft_test_loss_barrier_avg_vs_no_align": (
            soft["test_loss_barrier_avg"] - no_alignment["test_loss_barrier_avg"]
        ),
        "delta_soft_test_loss_barrier_max_endpoint_vs_no_align": (
            soft["test_loss_barrier_max_endpoint"] - no_alignment["test_loss_barrier_max_endpoint"]
        ),
        "delta_soft_min_test_acc_vs_no_align": soft["min_test_acc"] - no_alignment["min_test_acc"],
    }


def print_top_runs(rows: list[Dict[str, Any]], *, top_k: int = 5) -> None:
    """Print the best completed runs by hard barrier/mean-loss metrics."""

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
            f"steps={row['alignment_steps']} tau={row['tau']} lr={row['lr']} l={row['sinkhorn_l']} "
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
            f"steps={row['alignment_steps']} tau={row['tau']} lr={row['lr']} l={row['sinkhorn_l']} "
            f"hard_mean={row['hard_mean_test_interp_loss']:.4f} "
            f"hard_barrier={row['hard_test_loss_barrier_max_endpoint']:.4f} "
            f"hard_min_acc={row['hard_min_test_acc']:.2f}"
        )


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_rebasin_vgg16_sweep",
)
def main(cfg: DictConfig) -> None:
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

    base_output_root = Path(to_absolute_path(cfg.base_output_root))
    ensure_dir(base_output_root)

    sweep_summary: Dict[str, Any] = {
        "experiment_name": str(cfg.experiment_name),
        "base_output_root": str(base_output_root),
        "total_configured_runs": total_runs,
        "start_index": start_index,
        "end_index": end_index,
        "continue_on_error": continue_on_error,
        "runs": [],
    }

    print("=" * 80)
    print("EXTERNAL SINKHORN FULL SWEEP")
    print("=" * 80)
    print(f"total configured runs: {total_runs}")
    print(f"selected range: {start_index}..{end_index}")
    print(f"base_output_root: {base_output_root}")
    print("")

    for task_id in range(start_index, end_index + 1):
        combo = combos[task_id]
        output_tag = build_output_tag(combo)
        output_root = str(base_output_root / output_tag)
        experiment_name = f"{cfg.experiment_name}_{output_tag}"

        run_cfg = OmegaConf.create(
            {
                "experiment_name": experiment_name,
                "model_a_checkpoint": to_absolute_path(cfg.model_a_checkpoint),
                "model_b_checkpoint": to_absolute_path(cfg.model_b_checkpoint),
                "output_root": output_root,
                "data_path": to_absolute_path(cfg.data_path),
                "seed": int(cfg.seed),
                "num_workers": int(cfg.num_workers),
                "alignment_steps": int(combo["alignment_steps"]),
                "alignment_batch_size": int(cfg.alignment_batch_size),
                "calibration_size": int(cfg.calibration_size),
                "lr": float(combo["lr"]),
                "tau": float(combo["tau"]),
                "sinkhorn_iters": 20,
                "sinkhorn_l": float(combo["sinkhorn_l"]),
                "identity_init": bool(cfg.identity_init),
                "train_objective": str(cfg.train_objective),
                "midpoint_alpha": float(cfg.midpoint_alpha),
                "train_alpha_grid": [float(alpha) for alpha in cfg.train_alpha_grid],
                "device": str(cfg.device),
                "log_interval": int(cfg.log_interval),
                "log_eval_batches": int(cfg.log_eval_batches),
                "log_alpha_grid": [float(alpha) for alpha in cfg.log_alpha_grid],
                "evaluation_batch_size": int(cfg.evaluation_batch_size),
                "num_eval_points": int(cfg.num_eval_points),
                "plot_filename": str(cfg.plot_filename),
                "sweep_task_id": task_id,
                "sweep_total_runs": total_runs,
                "sweep_combo": combo,
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
            metadata = run_external_sinkhorn_baseline(run_cfg)
            run_record["status"] = "completed"
            run_record["metadata_path"] = metadata.get("metadata_path")
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
    print("EXTERNAL SINKHORN FULL SWEEP COMPLETE")
    print("=" * 80)
    print(f"Summary: {base_output_root / 'sweep_summary.json'}")
    if comparison_rows:
        print(f"Comparison table: {base_output_root / 'sweep_comparison.json'}")


if __name__ == "__main__":
    main()
