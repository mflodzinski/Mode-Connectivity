"""Run local git-rebasin-style weight matching on VGG16/MNIST checkpoints."""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict

project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".mplcache"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import hydra
import matplotlib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_global_seed
from scripts.lib.analysis.alignment import create_vgg16_model, load_dataset_eval_loaders
from scripts.lib.alignment.permutation_pipeline import (
    compute_barrier_metrics,
    evaluate_linear_interpolation,
    load_checkpoint_state_dict,
    resolve_device,
    save_checkpoint_with_state_dict,
    save_interpolation_results,
    write_summary_files,
)
from scripts.lib.alignment.permutation_spec import vgg16_permutation_spec
from scripts.lib.alignment.weight_matching import (
    apply_permutation as apply_permutation_torch,
    weight_matching,
)
from scripts.lib.core.output import ensure_dir, save_json


VARIANT_DISPLAY_NAMES = {
    "no_alignment": "No alignment",
    "git_rebasin_weight_matching": "Git Re-Basin weight matching",
}

VARIANT_STYLES = {
    "no_alignment": {"color": "#111827", "linestyle": "-"},
    "git_rebasin_weight_matching": {"color": "#059669", "linestyle": "-"},
}


def evaluate_endpoint_metrics(
    state_dict: Dict[str, torch.Tensor],
    loaders,
    *,
    device: torch.device,
) -> Dict[str, float]:
    model = create_vgg16_model(num_classes=10, device=device)
    model.load_state_dict(OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items()))
    model.eval()

    def _metrics(loader) -> Dict[str, float]:
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
                total_correct += logits.argmax(dim=1).eq(targets).sum().item()
                total_examples += targets.size(0)
        return {
            "loss": total_loss / total_examples,
            "accuracy": 100.0 * total_correct / total_examples,
        }

    train_metrics = _metrics(loaders["train"])
    test_metrics = _metrics(loaders["test"])
    return {
        "train_loss": train_metrics["loss"],
        "train_acc": train_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_acc": test_metrics["accuracy"],
    }


def plot_variant_curves(output_path: str, variant_results: Dict[str, Dict[str, np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    metric_layout = [
        ("tr_loss", "Train Loss", axes[0, 0]),
        ("te_loss", "Test Loss", axes[0, 1]),
        ("tr_acc", "Train Accuracy", axes[1, 0]),
        ("te_acc", "Test Accuracy", axes[1, 1]),
    ]

    for variant_key, results in variant_results.items():
        style = VARIANT_STYLES[variant_key]
        for metric_name, title, axis in metric_layout:
            axis.plot(
                results["ts"],
                results[metric_name],
                label=VARIANT_DISPLAY_NAMES[variant_key],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )
            axis.set_title(title)
            axis.set_xlabel("Interpolation t")
            axis.grid(True, alpha=0.25)

    axes[0, 0].legend(loc="upper center", bbox_to_anchor=(1.05, 1.35), ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="git_rebasin_vgg16_mnist_weight_matching",
)
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))

    runtime_device = resolve_device(str(cfg.device))
    output_root = Path(to_absolute_path(cfg.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    model_a_checkpoint = to_absolute_path(cfg.model_a_checkpoint)
    model_b_checkpoint = to_absolute_path(cfg.model_b_checkpoint)
    data_path = to_absolute_path(cfg.data_path)

    state_a = load_checkpoint_state_dict(model_a_checkpoint)
    state_b = load_checkpoint_state_dict(model_b_checkpoint)

    ps = vgg16_permutation_spec()
    params_a = OrderedDict((key, value.detach().cpu()) for key, value in state_a.items())
    params_b = OrderedDict((key, value.detach().cpu()) for key, value in state_b.items())

    print("=" * 80)
    print("GIT-REBASIN WEIGHT MATCHING (VGG16/MNIST)")
    print("=" * 80)
    print(f"experiment_name: {cfg.experiment_name}")
    print(f"model_a_checkpoint: {model_a_checkpoint}")
    print(f"model_b_checkpoint: {model_b_checkpoint}")
    print(f"output_root: {output_root}")
    print(f"dataset: {cfg.dataset}")
    print(f"device: {runtime_device}")
    print(f"max_iter: {cfg.max_iter}")
    print("")

    permutation_np = weight_matching(
        ps,
        params_a,
        params_b,
        max_iter=int(cfg.max_iter),
        seed=int(cfg.seed),
        silent=False,
    )
    aligned_state_b = apply_permutation_torch(ps, permutation_np, state_b)

    aligned_checkpoint_path = str(output_root / "aligned.pt")
    permutation_path = str(output_root / "permutation.json")
    metadata_path = str(output_root / "metadata.json")

    save_checkpoint_with_state_dict(
        model_b_checkpoint,
        aligned_checkpoint_path,
        aligned_state_b,
        metadata={
            "method": "git_rebasin_weight_matching",
            "permutation_path": permutation_path,
            "metadata_path": metadata_path,
        },
    )
    save_json({name: values.tolist() for name, values in permutation_np.items()}, permutation_path, indent=2)

    loaders, _ = load_dataset_eval_loaders(
        dataset=str(cfg.dataset),
        data_path=data_path,
        batch_size=int(cfg.evaluation_batch_size),
        num_workers=int(cfg.num_workers),
    )

    endpoint_a = evaluate_endpoint_metrics(state_a, loaders, device=runtime_device)
    variants = {
        "no_alignment": state_b,
        "git_rebasin_weight_matching": aligned_state_b,
    }

    evaluation_dir = ensure_dir(output_root / "evaluation")
    variant_rows = []
    variant_results: Dict[str, Dict[str, np.ndarray]] = {}
    for variant_key, variant_state in variants.items():
        interpolation = evaluate_linear_interpolation(
            state_a,
            variant_state,
            loaders,
            num_points=int(cfg.num_eval_points),
            device=runtime_device,
        )
        variant_dir = ensure_dir(Path(evaluation_dir) / variant_key)
        save_interpolation_results(str(Path(variant_dir) / "interpolation.npz"), interpolation)
        variant_results[variant_key] = interpolation

        endpoint_b = evaluate_endpoint_metrics(variant_state, loaders, device=runtime_device)
        barriers = compute_barrier_metrics(interpolation)
        variant_rows.append(
            {
                "variant_key": variant_key,
                "variant_name": VARIANT_DISPLAY_NAMES[variant_key],
                "endpoint_a_test_loss": endpoint_a["test_loss"],
                "endpoint_a_test_acc": endpoint_a["test_acc"],
                "endpoint_b_test_loss": endpoint_b["test_loss"],
                "endpoint_b_test_acc": endpoint_b["test_acc"],
                "mean_train_interp_loss": float(np.mean(interpolation["tr_loss"])),
                "mean_test_interp_loss": float(np.mean(interpolation["te_loss"])),
                "raw_max_train_interp_loss": float(np.max(interpolation["tr_loss"])),
                "raw_max_test_interp_loss": float(np.max(interpolation["te_loss"])),
                "train_loss_barrier_avg": barriers["train_loss_barrier_avg"],
                "test_loss_barrier_avg": barriers["test_loss_barrier_avg"],
                "train_loss_barrier_max_endpoint": barriers["train_loss_barrier_max_endpoint"],
                "test_loss_barrier_max_endpoint": barriers["test_loss_barrier_max_endpoint"],
                "min_train_acc": barriers["min_train_acc"],
                "min_test_acc": barriers["min_test_acc"],
                "train_acc_drop_from_endpoint_min": barriers["train_acc_drop_from_endpoint_min"],
                "test_acc_drop_from_endpoint_min": barriers["test_acc_drop_from_endpoint_min"],
            }
        )

    plot_path = str(Path(evaluation_dir) / str(cfg.plot_filename))
    plot_variant_curves(plot_path, variant_results)
    write_summary_files(str(evaluation_dir), variant_rows)

    metadata = {
        "experiment_name": str(cfg.experiment_name),
        "model_a_checkpoint": model_a_checkpoint,
        "model_b_checkpoint": model_b_checkpoint,
        "output_root": str(output_root),
        "aligned_checkpoint_path": aligned_checkpoint_path,
        "permutation_path": permutation_path,
        "evaluation_dir": str(evaluation_dir),
        "plot_path": plot_path,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(metadata, metadata_path, indent=2)

    print("")
    print("=" * 80)
    print("GIT-REBASIN WEIGHT MATCHING COMPLETE")
    print("=" * 80)
    print(f"Aligned checkpoint: {aligned_checkpoint_path}")
    print(f"Permutation: {permutation_path}")
    print(f"Evaluation summary: {Path(evaluation_dir) / 'comparison.json'}")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
