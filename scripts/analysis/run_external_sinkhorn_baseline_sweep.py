"""Run one task from the configured external Sinkhorn sweep.

This script reads the sweep grid from the Hydra config, maps the current
``SLURM_ARRAY_TASK_ID`` to one hyperparameter tuple, and dispatches the actual
experiment run through ``run_external_sinkhorn_baseline``.
"""

from __future__ import annotations

import os
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_baseline import run_external_sinkhorn_baseline


def _listify(values: Iterable[Any]) -> list[Any]:
    return [OmegaConf.to_container(value, resolve=True) if isinstance(value, DictConfig) else value for value in values]


def enumerate_sweep_combinations(sweep_cfg: DictConfig) -> List[Dict[str, Any]]:
    """Enumerate the Cartesian product defined by the sweep config."""

    keys = list(sweep_cfg.keys())
    value_lists = [_listify(sweep_cfg[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*value_lists)]


def sanitize_value(value: Any) -> str:
    text = str(value)
    text = text.replace(".", "p")
    text = text.replace("-", "_")
    return text


def build_output_tag(combo: Dict[str, Any]) -> str:
    return "_".join(
        [
            f"steps{combo['alignment_steps']}",
            f"tau{sanitize_value(combo['tau'])}",
            f"lr{sanitize_value(combo['lr'])}",
            f"l{sanitize_value(combo['sinkhorn_l'])}",
        ]
    )


def resolve_task_id(cfg: DictConfig) -> int:
    if "sweep_task_id" in cfg and cfg.sweep_task_id is not None:
        return int(cfg.sweep_task_id)
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        return int(os.environ["SLURM_ARRAY_TASK_ID"])
    if "SWEEP_TASK_ID" in os.environ:
        return int(os.environ["SWEEP_TASK_ID"])
    return 0


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_rebasin_vgg16_sweep",
)
def main(cfg: DictConfig) -> None:
    combos = enumerate_sweep_combinations(cfg.sweep)
    task_id = resolve_task_id(cfg)
    total_runs = len(combos)

    if task_id < 0 or task_id >= total_runs:
        raise ValueError(f"Sweep task id {task_id} is out of range for {total_runs} configured runs.")

    combo = combos[task_id]
    output_tag = build_output_tag(combo)
    output_root = str(Path(to_absolute_path(cfg.base_output_root)) / output_tag)
    experiment_name = f"{cfg.experiment_name}_{output_tag}"

    run_cfg = OmegaConf.create(
        {
            "experiment_name": experiment_name,
            "model_a_checkpoint": to_absolute_path(cfg.model_a_checkpoint),
            "model_b_checkpoint": to_absolute_path(cfg.model_b_checkpoint),
            "output_root": output_root,
            "dataset": str(cfg.dataset),
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

    print("=" * 80)
    print("EXTERNAL SINKHORN SWEEP TASK")
    print("=" * 80)
    print(f"task_id: {task_id}/{total_runs}")
    print(f"experiment_name: {experiment_name}")
    print(f"output_root: {output_root}")
    print(f"combo: {combo}")
    print("")

    run_external_sinkhorn_baseline(run_cfg)


if __name__ == "__main__":
    main()
