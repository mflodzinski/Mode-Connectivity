"""Run the Sinkhorn experiment matrix across 3 endpoint pairs for one mode."""

from __future__ import annotations

import itertools
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import import_original_mnist_components
from scripts.analysis.run_external_sinkhorn_original_vgg_cifar10_align_sweep_all import (
    build_cifar10_loaders,
    run_alignment_sweep_all as run_cifar_alignment_sweep_all,
)
from scripts.analysis.run_external_sinkhorn_original_vgg_mnist_align_sweep_all import (
    build_mnist_loaders,
    run_alignment_sweep_all as run_mnist_alignment_sweep_all,
)
from scripts.analysis.sinkhorn_experiment_utils import (
    evaluate_interpolation_results,
    load_model_from_checkpoint,
    save_interpolation_npz,
)
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir, load_json, save_json


def sanitize_identifier(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()


def dataset_slug(dataset: str) -> str:
    return sanitize_identifier(str(dataset))


def model_slug(vgg_name: str) -> str:
    return sanitize_identifier(str(vgg_name))


def mode_slug(mode_name: str) -> str:
    return sanitize_identifier(str(mode_name))


def validate_endpoint_models(cfg: DictConfig) -> list[dict[str, str]]:
    endpoints = []
    raw_endpoints = list(cfg.get("endpoint_models", []))
    if len(raw_endpoints) != 3:
        raise ValueError(f"Expected exactly 3 endpoint_models entries, got {len(raw_endpoints)}.")

    seen_names: set[str] = set()
    for entry in raw_endpoints:
        name = str(entry.get("name", "")).strip()
        checkpoint = entry.get("checkpoint", None)
        if not name:
            raise ValueError("Each endpoint_models entry must define a non-empty 'name'.")
        if name in seen_names:
            raise ValueError(f"Duplicate endpoint model name {name!r}.")
        if checkpoint in (None, "", "null"):
            raise ValueError(f"Endpoint model {name!r} is missing a checkpoint path.")
        seen_names.add(name)
        endpoints.append(
            {
                "name": name,
                "checkpoint": str(checkpoint),
            }
        )
    return endpoints


def derive_pairs(endpoint_models: list[dict[str, str]]) -> list[dict[str, Any]]:
    pairs = []
    for left, right in itertools.combinations(endpoint_models, 2):
        pair_id = f"{sanitize_identifier(left['name'])}-{sanitize_identifier(right['name'])}"
        pairs.append(
            {
                "pair_id": pair_id,
                "model_a_name": left["name"],
                "model_b_name": right["name"],
                "model_a_checkpoint": left["checkpoint"],
                "model_b_checkpoint": right["checkpoint"],
            }
        )
    return pairs


def get_mode_root(cfg: DictConfig) -> Path:
    return ensure_dir(
        Path(to_absolute_path(str(cfg.base_output_root)))
        / dataset_slug(str(cfg.dataset))
        / model_slug(str(cfg.vgg_name))
        / mode_slug(str(cfg.mode_name))
    )


def get_pair_root(cfg: DictConfig, pair_id: str) -> Path:
    return ensure_dir(get_mode_root(cfg) / f"pair_{pair_id}")


def get_permutation_only_root(cfg: DictConfig) -> Path:
    explicit_root = cfg.get("permutation_only_root", None)
    if explicit_root not in (None, "", "null"):
        return Path(to_absolute_path(str(explicit_root)))
    return (
        Path(to_absolute_path(str(cfg.base_output_root)))
        / dataset_slug(str(cfg.dataset))
        / model_slug(str(cfg.vgg_name))
        / "permutation-only"
    )


def get_selected_dir(pair_root: Path, selector_name: str) -> Path:
    return ensure_dir(pair_root / "selected" / selector_name)


def get_permutation_warm_start_artifact(cfg: DictConfig, pair_id: str) -> Path:
    selector_name = "best_by_val_loss"
    if str(cfg.get("warm_start_selection", "val_loss")).lower() == "val_barrier":
        selector_name = "best_by_val_barrier"
    artifact_path = get_permutation_only_root(cfg) / f"pair_{pair_id}" / "selected" / selector_name / "alignment_artifacts.pt"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Required permutation-only warm start artifact not found: {artifact_path}. "
            "Run permutation_only first for the same dataset/model/pair."
        )
    return artifact_path


def build_pair_cfg(cfg: DictConfig, pair: dict[str, Any], pair_root: Path, starting_alignment_artifact: str | None) -> DictConfig:
    return OmegaConf.create(
        {
            "experiment_name": str(cfg.experiment_name),
            "dataset": str(cfg.dataset),
            "vgg_name": str(cfg.vgg_name),
            "data_path": str(cfg.data_path),
            "seed": int(cfg.seed),
            "split_seed": int(cfg.split_seed),
            "num_workers": int(cfg.num_workers),
            "val_fraction": float(cfg.val_fraction),
            "batch_size": int(cfg.batch_size),
            "image_size": int(cfg.image_size),
            "model_a_checkpoint": str(pair["model_a_checkpoint"]),
            "model_b_checkpoint": str(pair["model_b_checkpoint"]),
            "model_a_name": str(pair["model_a_name"]),
            "model_b_name": str(pair["model_b_name"]),
            "pair_id": str(pair["pair_id"]),
            "base_output_root": str(pair_root / "sweeps"),
            "sinkhorn_iters": int(cfg.sinkhorn_iters),
            "identity_init": bool(cfg.identity_init),
            "scale_invariant": bool(cfg.scale_invariant),
            "lambda_scale": float(cfg.get("lambda_scale", 0.0)),
            "finetune_mode": str(cfg.finetune_mode),
            "best_eval_interval": int(cfg.best_eval_interval),
            "validation_alpha_grid": list(cfg.validation_alpha_grid),
            "starting_alignment_artifact": starting_alignment_artifact,
            "starting_permutation_kind": str(cfg.get("starting_permutation_kind", "raw")),
            "num_eval_points": int(cfg.num_eval_points),
            "device": str(cfg.device),
            "log_interval": int(cfg.log_interval),
            "continue_on_error": bool(cfg.get("continue_on_error", True)),
            "mode_name": str(cfg.mode_name),
            "sweep": OmegaConf.to_container(cfg.sweep, resolve=True),
        }
    )


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def load_resolved_run_config(run_dir: Path) -> DictConfig:
    return OmegaConf.load(run_dir / "resolved_config.yaml")


def export_selected_interpolations(run_dir: Path, selection_dir: Path) -> dict[str, Any]:
    run_cfg = load_resolved_run_config(run_dir)
    VGGClass, _, _, dnn_data, _, _, lerp = import_original_mnist_components()
    from utils import eval_loss_acc

    device = resolve_device(str(run_cfg.device))
    dataset_name = str(run_cfg.get("dataset_name", run_cfg.get("dataset", "CIFAR10"))).upper()
    if dataset_name == "CIFAR10":
        train_loader, _, test_loader = build_cifar10_loaders(run_cfg, dnn_data)
    elif dataset_name == "MNIST":
        train_loader, _, test_loader = build_mnist_loaders(run_cfg, dnn_data)
    else:
        raise ValueError(f"Unsupported dataset_name={dataset_name!r} in resolved run config.")
    dataset_info = {
        "dataset": dataset_name,
        "num_eval_points": int(run_cfg.num_eval_points),
    }
    model_a = load_model_from_checkpoint(
        Path(run_cfg.model_a_checkpoint),
        VGGClass,
        vgg_name=str(run_cfg.vgg_name),
        image_size=int(run_cfg.image_size),
        device=device,
    )
    model_b = load_model_from_checkpoint(
        Path(run_cfg.model_b_checkpoint),
        VGGClass,
        vgg_name=str(run_cfg.vgg_name),
        image_size=int(run_cfg.image_size),
        device=device,
    )
    rebased = load_model_from_checkpoint(
        run_dir / "rebased_model.pt",
        VGGClass,
        vgg_name=str(run_cfg.vgg_name),
        image_size=int(run_cfg.image_size),
        device=device,
    )

    old_results = evaluate_interpolation_results(
        model_a,
        model_b,
        train_loader=train_loader,
        test_loader=test_loader,
        lerp=lerp,
        eval_loss_acc=eval_loss_acc,
        device=device,
        num_eval_points=int(run_cfg.num_eval_points),
    )
    new_results = evaluate_interpolation_results(
        rebased,
        model_b,
        train_loader=train_loader,
        test_loader=test_loader,
        lerp=lerp,
        eval_loss_acc=eval_loss_acc,
        device=device,
        num_eval_points=int(run_cfg.num_eval_points),
    )
    save_interpolation_npz(selection_dir / "old_interpolation.npz", old_results)
    save_interpolation_npz(selection_dir / "new_interpolation.npz", new_results)

    summary = {
        "dataset_info": dataset_info,
        "old_interpolation_npz": str(selection_dir / "old_interpolation.npz"),
        "new_interpolation_npz": str(selection_dir / "new_interpolation.npz"),
    }
    save_json(summary, selection_dir / "interpolation_exports.json", indent=2)
    return summary


def write_selection_artifacts(pair_root: Path, selector_name: str, row: dict[str, Any]) -> dict[str, Any]:
    selection_dir = get_selected_dir(pair_root, selector_name)
    run_dir = Path(row["output_root"])

    save_json(row, selection_dir / "selection_summary.json", indent=2)
    copy_if_exists(run_dir / "resolved_config.yaml", selection_dir / "resolved_config.yaml")
    copy_if_exists(run_dir / "comparison.json", selection_dir / "comparison.json")
    copy_if_exists(run_dir / "metadata.json", selection_dir / "metadata.json")
    copy_if_exists(run_dir / "alignment_artifacts.pt", selection_dir / "alignment_artifacts.pt")

    exported = {"selection_dir": str(selection_dir), "selected_run_dir": str(run_dir)}
    if selector_name == "best_by_val_loss":
        copy_if_exists(run_dir / "rebased_model.pt", selection_dir / "rebased_model.pt")
        exported.update(export_selected_interpolations(run_dir, selection_dir))

    save_json(exported, selection_dir / "export_summary.json", indent=2)
    return exported


def select_best_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    completed_rows = [row for row in rows if row.get("best_alignment_score") is not None]
    if not completed_rows:
        raise ValueError("No completed sweep rows available for selection.")

    best_by_val_loss = min(completed_rows, key=lambda row: float(row["best_alignment_score"]))
    best_by_val_barrier = min(completed_rows, key=lambda row: float(row["lmc_val_loss_barrier_max_endpoint"]))
    return best_by_val_loss, best_by_val_barrier


def run_matrix(cfg: DictConfig) -> dict[str, Any]:
    endpoint_models = validate_endpoint_models(cfg)
    pairs = derive_pairs(endpoint_models)
    mode_root = get_mode_root(cfg)

    print("")
    print("=" * 80)
    print("SINKHORN EXPERIMENT MATRIX")
    print("=" * 80)
    print(f"dataset: {cfg.dataset}")
    print(f"model: {cfg.vgg_name}")
    print(f"mode_name: {cfg.mode_name}")
    print(f"mode_root: {mode_root}")
    print(f"requires_permutation_warm_start: {bool(cfg.get('requires_permutation_warm_start', False))}")
    print("")

    pair_summaries: list[dict[str, Any]] = []
    for pair in pairs:
        pair_root = get_pair_root(cfg, pair["pair_id"])
        starting_alignment_artifact = None
        if bool(cfg.get("requires_permutation_warm_start", False)):
            starting_alignment_artifact = str(get_permutation_warm_start_artifact(cfg, pair["pair_id"]))

        pair_cfg = build_pair_cfg(cfg, pair, pair_root, starting_alignment_artifact)
        sweeps_root = Path(pair_cfg.base_output_root)
        print("-" * 80)
        print(f"pair_id: {pair['pair_id']}")
        print(f"model_a: {pair['model_a_name']} -> {pair['model_a_checkpoint']}")
        print(f"model_b: {pair['model_b_name']} -> {pair['model_b_checkpoint']}")
        if starting_alignment_artifact is not None:
            print(f"warm_start_artifact: {starting_alignment_artifact}")
        print("")

        dataset_name = str(cfg.dataset).upper()
        if dataset_name == "CIFAR10":
            run_cifar_alignment_sweep_all(pair_cfg)
        elif dataset_name == "MNIST":
            run_mnist_alignment_sweep_all(pair_cfg)
        else:
            raise ValueError(f"Unsupported dataset={cfg.dataset!r}.")

        comparison_rows = load_json(sweeps_root / "sweep_comparison.json")
        best_by_val_loss, best_by_val_barrier = select_best_rows(comparison_rows)
        val_loss_export = write_selection_artifacts(pair_root, "best_by_val_loss", best_by_val_loss)
        val_barrier_export = write_selection_artifacts(pair_root, "best_by_val_barrier", best_by_val_barrier)

        pair_summary = {
            "pair_id": pair["pair_id"],
            "model_a_name": pair["model_a_name"],
            "model_b_name": pair["model_b_name"],
            "model_a_checkpoint": pair["model_a_checkpoint"],
            "model_b_checkpoint": pair["model_b_checkpoint"],
            "warm_start_artifact": starting_alignment_artifact,
            "sweeps_root": str(sweeps_root),
            "best_by_val_loss": {
                "row": best_by_val_loss,
                "exports": val_loss_export,
            },
            "best_by_val_barrier": {
                "row": best_by_val_barrier,
                "exports": val_barrier_export,
            },
        }
        save_json(pair_summary, pair_root / "pair_summary.json", indent=2)
        pair_summaries.append(pair_summary)

    matrix_summary = {
        "dataset": str(cfg.dataset),
        "vgg_name": str(cfg.vgg_name),
        "mode_name": str(cfg.mode_name),
        "mode_root": str(mode_root),
        "pair_summaries": pair_summaries,
    }
    save_json(matrix_summary, mode_root / "matrix_summary.json", indent=2)
    print("")
    print("=" * 80)
    print("SINKHORN EXPERIMENT MATRIX COMPLETE")
    print("=" * 80)
    print(f"Results written under: {mode_root}")
    return matrix_summary


@hydra.main(version_base=None, config_path="../../configs/analysis", config_name="external_sinkhorn_experiment_matrix_permutation_only")
def main(cfg: DictConfig) -> None:
    run_matrix(cfg)


if __name__ == "__main__":
    main()
