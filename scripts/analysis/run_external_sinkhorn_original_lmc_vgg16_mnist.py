"""Run the original external/sinkhorn-rebasin LMC loop on VGG16/MNIST endpoints.

This entrypoint stays close to the original repo examples:

- load two already-trained endpoint models A and B
- wrap model A in ``RebasinNet``
- optimize the original ``MidLoss`` or ``RndLoss`` objective against model B
- switch to ``eval()`` to obtain the hard projected alignment

The only adaptation is checkpoint translation between this repo's local VGG16
layout and the vendored external VGG16 layout used by ``sinkhorn-rebasin``.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping

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
from scripts.analysis.run_external_sinkhorn_baseline import (
    build_calibration_loader,
    build_external_vgg,
    evaluate_endpoint_metrics,
    external_to_native_state_dict,
    import_external_sinkhorn,
    validate_state_translation,
)
from scripts.lib.analysis.alignment import load_dataset_eval_loaders
from scripts.lib.alignment.permutation_pipeline import (
    compute_barrier_metrics,
    evaluate_linear_interpolation,
    load_checkpoint_state_dict,
    resolve_device,
    save_checkpoint_with_state_dict,
    save_interpolation_results,
    write_summary_files,
)
from scripts.lib.core.output import ensure_dir, save_json


VARIANT_DISPLAY_NAMES = {
    "no_alignment": "No alignment",
    "original_sinkhorn_soft": "Original Sinkhorn LMC (soft)",
    "original_sinkhorn_hard": "Original Sinkhorn LMC (hard)",
}

VARIANT_STYLES = {
    "no_alignment": {"color": "#111827", "linestyle": "-"},
    "original_sinkhorn_soft": {"color": "#dc2626", "linestyle": "-"},
    "original_sinkhorn_hard": {"color": "#2563eb", "linestyle": "--"},
}


def import_original_lmc_components():
    """Import the vendored RebasinNet API used in the original examples."""

    _, RebasinNet, matching = import_external_sinkhorn()
    from rebasin.loss import MidLoss, RndLoss, DistL1Loss, DistL2Loss, DistCosineLoss

    return RebasinNet, matching, MidLoss, RndLoss, DistL1Loss, DistL2Loss, DistCosineLoss


def snapshot_rebasin_states(rebasin_net) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    """Capture both the soft-training and hard-eval aligned states."""

    was_training = rebasin_net.training

    rebasin_net.train()
    soft_model = rebasin_net()
    soft_state = OrderedDict((key, value.detach().cpu().clone()) for key, value in soft_model.state_dict().items())

    rebasin_net.eval()
    hard_model = rebasin_net()
    hard_state = OrderedDict((key, value.detach().cpu().clone()) for key, value in hard_model.state_dict().items())

    rebasin_net.train(was_training)
    return soft_state, hard_state


def evaluate_epoch_objective(
    rebasin_net,
    criterion,
    loader,
    *,
    device: torch.device,
    requires_data: bool,
) -> float:
    """Evaluate the original LMC loss over one loader pass."""

    if not requires_data:
        with torch.no_grad():
            rebased_model = rebasin_net()
            return float(criterion(rebased_model).item())

    total_loss = 0.0
    total_examples = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            rebased_model = rebasin_net()
            loss = criterion(rebased_model, inputs, targets)
            total_loss += loss.item() * targets.size(0)
            total_examples += targets.size(0)

    if total_examples == 0:
        raise ValueError("Calibration loader produced zero examples.")
    return total_loss / total_examples


def plot_variant_curves(output_path: str, variant_results: Mapping[str, Dict[str, np.ndarray]]) -> None:
    """Write a compact interpolation plot for no-alignment vs Sinkhorn LMC."""

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


def run_original_sinkhorn_lmc_vgg16_mnist(cfg: DictConfig | Mapping[str, Any]) -> Dict[str, Any]:
    """Run the original external Sinkhorn LMC loop on local VGG16/MNIST checkpoints."""

    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))

    set_global_seed(int(cfg.seed))
    runtime_device = resolve_device(str(cfg.device))
    output_root = Path(to_absolute_path(cfg.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    model_a_checkpoint = to_absolute_path(str(cfg.model_a_checkpoint))
    model_b_checkpoint = to_absolute_path(str(cfg.model_b_checkpoint))
    dataset = str(cfg.dataset)
    data_path = to_absolute_path(str(cfg.data_path))

    state_a_native = load_checkpoint_state_dict(model_a_checkpoint)
    state_b_native = load_checkpoint_state_dict(model_b_checkpoint)

    model_a_external = build_external_vgg(state_a_native, device=runtime_device)
    model_b_external = build_external_vgg(state_b_native, device=runtime_device)
    translation_checks = validate_state_translation(state_a_native, model_a_external, device=runtime_device)

    RebasinNet, matching, MidLoss, RndLoss, DistL1Loss, DistL2Loss, DistCosineLoss = import_original_lmc_components()
    rebasin_net = RebasinNet(
        model_a_external,
        input_shape=(1, 3, 32, 32),
        l=float(cfg.sinkhorn_l),
        tau=float(cfg.tau),
        n_iter=int(cfg.sinkhorn_iters),
    )
    rebasin_net.to(runtime_device)
    if bool(cfg.identity_init):
        rebasin_net.identity_init()

    loss_name = str(cfg.loss_name).lower()
    if loss_name == "midpoint":
        criterion = MidLoss(model_b_external, criterion=torch.nn.CrossEntropyLoss())
        requires_data = True
    elif loss_name == "random":
        criterion = RndLoss(model_b_external, criterion=torch.nn.CrossEntropyLoss())
        requires_data = True
    elif loss_name == "dist_l2":
        criterion = DistL2Loss(model_b_external)
        requires_data = False
    elif loss_name == "dist_l1":
        criterion = DistL1Loss(model_b_external)
        requires_data = False
    elif loss_name == "dist_cosine":
        criterion = DistCosineLoss(model_b_external)
        requires_data = False
    else:
        raise ValueError(
            f"Unsupported loss_name={cfg.loss_name!r}. "
            "Expected one of 'midpoint', 'random', 'dist_l1', 'dist_l2', 'dist_cosine'."
        )

    optimizer = torch.optim.AdamW(rebasin_net.p.parameters(), lr=float(cfg.lr))
    calibration_loader = build_calibration_loader(
        dataset=dataset,
        data_path=data_path,
        calibration_size=int(cfg.calibration_size),
        batch_size=int(cfg.alignment_batch_size),
        num_workers=int(cfg.num_workers),
        seed=int(cfg.seed),
    )

    print("=" * 80)
    print("ORIGINAL EXTERNAL SINKHORN LMC")
    print("=" * 80)
    print(f"experiment_name: {cfg.experiment_name}")
    print(f"model_a_checkpoint: {model_a_checkpoint}")
    print(f"model_b_checkpoint: {model_b_checkpoint}")
    print(f"output_root: {output_root}")
    print(f"dataset: {dataset}")
    print(f"loss_name: {cfg.loss_name}")
    print(f"device: {runtime_device}")
    print("")

    history: list[Dict[str, float | int]] = []
    for epoch in range(1, int(cfg.alignment_epochs) + 1):
        rebasin_net.train()
        if requires_data:
            cumulative_train_loss = 0.0
            total_examples = 0
            for inputs, targets in calibration_loader:
                inputs = inputs.to(runtime_device)
                targets = targets.to(runtime_device)
                rebased_model = rebasin_net()
                loss = criterion(rebased_model, inputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                cumulative_train_loss += loss.item() * targets.size(0)
                total_examples += targets.size(0)

            if total_examples == 0:
                raise ValueError("Calibration loader produced zero examples.")
            train_loss = cumulative_train_loss / total_examples
        else:
            rebased_model = rebasin_net()
            loss = criterion(rebased_model)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = float(loss.item())

        rebasin_net.eval()
        hard_loss = evaluate_epoch_objective(
            rebasin_net,
            criterion,
            calibration_loader,
            device=runtime_device,
            requires_data=requires_data,
        )

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "hard_loss": float(hard_loss),
        }
        history.append(epoch_metrics)

        should_log = epoch == 1 or epoch == int(cfg.alignment_epochs) or (
            int(cfg.log_interval) > 0 and epoch % int(cfg.log_interval) == 0
        )
        if should_log:
            print(
                f"[original_sinkhorn_lmc] epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"hard_loss={hard_loss:.4f}"
            )

    soft_state_external, hard_state_external = snapshot_rebasin_states(rebasin_net)
    soft_state_native = external_to_native_state_dict(soft_state_external)
    hard_state_native = external_to_native_state_dict(hard_state_external)

    soft_checkpoint_path = str(output_root / "soft_aligned.pt")
    hard_checkpoint_path = str(output_root / "hard_aligned.pt")
    artifact_path = str(output_root / "alignment_artifacts.pt")
    history_path = str(output_root / "training_history.json")
    metadata_path = str(output_root / "metadata.json")

    raw_permutation_parameters = [parameter.detach().cpu().clone() for parameter in rebasin_net.p if parameter is not None]
    hard_permutation_matrices = [
        matching(parameter.detach().cpu().numpy()).to(torch.float32).cpu()
        for parameter in rebasin_net.p
        if parameter is not None
    ]
    torch.save(
        {
            "raw_parameters": raw_permutation_parameters,
            "hard_permutations": hard_permutation_matrices,
            "translation_checks": translation_checks,
            "config": OmegaConf.to_container(cfg, resolve=True),
        },
        artifact_path,
    )

    save_checkpoint_with_state_dict(
        model_a_checkpoint,
        soft_checkpoint_path,
        soft_state_native,
        metadata={"method": "original_external_sinkhorn_lmc", "artifact_path": artifact_path},
    )
    save_checkpoint_with_state_dict(
        model_a_checkpoint,
        hard_checkpoint_path,
        hard_state_native,
        metadata={"method": "original_external_sinkhorn_lmc", "artifact_path": artifact_path},
    )
    save_json(history, history_path, indent=2)

    loaders, _ = load_dataset_eval_loaders(
        dataset=dataset,
        data_path=data_path,
        batch_size=int(cfg.evaluation_batch_size),
        num_workers=int(cfg.num_workers),
    )
    max_eval_batches = cfg.get("max_eval_batches", None)
    variants = {
        "no_alignment": {
            "state_a": state_a_native,
            "state_b": state_b_native,
        },
        "original_sinkhorn_soft": {
            "state_a": soft_state_native,
            "state_b": state_b_native,
        },
        "original_sinkhorn_hard": {
            "state_a": hard_state_native,
            "state_b": state_b_native,
        },
    }

    evaluation_dir = ensure_dir(output_root / "evaluation")
    variant_rows = []
    variant_results = {}
    for variant_key, variant_pair in variants.items():
        variant_state_a = variant_pair["state_a"]
        variant_state_b = variant_pair["state_b"]
        interpolation = evaluate_linear_interpolation(
            variant_state_a,
            variant_state_b,
            loaders,
            num_points=int(cfg.num_eval_points),
            device=runtime_device,
        )
        variant_dir = ensure_dir(Path(evaluation_dir) / variant_key)
        save_interpolation_results(str(Path(variant_dir) / "interpolation.npz"), interpolation)
        variant_results[variant_key] = interpolation

        endpoint_a = evaluate_endpoint_metrics(
            variant_state_a,
            loaders,
            device=runtime_device,
            max_eval_batches=max_eval_batches,
        )
        endpoint_b = evaluate_endpoint_metrics(
            variant_state_b,
            loaders,
            device=runtime_device,
            max_eval_batches=max_eval_batches,
        )
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
        "soft_checkpoint_path": soft_checkpoint_path,
        "hard_checkpoint_path": hard_checkpoint_path,
        "artifact_path": artifact_path,
        "history_path": history_path,
        "evaluation_dir": str(evaluation_dir),
        "plot_path": plot_path,
        "translation_checks": translation_checks,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(metadata, metadata_path, indent=2)

    print("")
    print("=" * 80)
    print("ORIGINAL EXTERNAL SINKHORN LMC RUN COMPLETE")
    print("=" * 80)
    print(f"Soft checkpoint: {soft_checkpoint_path}")
    print(f"Hard checkpoint: {hard_checkpoint_path}")
    print(f"Artifacts: {artifact_path}")
    print(f"Evaluation summary: {Path(evaluation_dir) / 'comparison.json'}")
    print(f"Comparison plot: {plot_path}")

    return metadata


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_lmc_vgg16_mnist",
)
def main(cfg: DictConfig) -> None:
    run_original_sinkhorn_lmc_vgg16_mnist(cfg)


if __name__ == "__main__":
    main()
