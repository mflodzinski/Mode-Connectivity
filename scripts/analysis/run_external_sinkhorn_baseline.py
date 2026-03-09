"""Run the vendored sinkhorn-rebasin baseline on VGG16/CIFAR10 endpoints.

This wrapper uses the implementation in ``external/sinkhorn-rebasin`` directly
and adapts the checkpoint format used in this repo to the external VGG layout.
It trains the external Sinkhorn baseline on model B against model A, saves the
soft and hard aligned checkpoints back in the native checkpoint format, and
writes a small interpolation comparison against the no-alignment baseline.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping

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
from torch.utils.data import DataLoader, Subset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils import set_global_seed
from scripts.lib.analysis.alignment import create_vgg16_model, evaluate_model, load_cifar10_eval_loaders
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


CONV_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("layer_blocks.0.0", "features.0"),
    ("layer_blocks.0.1", "features.2"),
    ("layer_blocks.1.0", "features.5"),
    ("layer_blocks.1.1", "features.7"),
    ("layer_blocks.2.0", "features.10"),
    ("layer_blocks.2.1", "features.12"),
    ("layer_blocks.2.2", "features.14"),
    ("layer_blocks.3.0", "features.17"),
    ("layer_blocks.3.1", "features.19"),
    ("layer_blocks.3.2", "features.21"),
    ("layer_blocks.4.0", "features.24"),
    ("layer_blocks.4.1", "features.26"),
    ("layer_blocks.4.2", "features.28"),
)

LINEAR_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("classifier.1", "classifier.1"),
    ("classifier.4", "classifier.4"),
    ("classifier.6", "classifier.6"),
)

VARIANT_DISPLAY_NAMES = {
    "no_alignment": "No alignment",
    "external_sinkhorn_soft": "External Sinkhorn (soft)",
    "external_sinkhorn_hard": "External Sinkhorn (hard)",
}

VARIANT_STYLES = {
    "no_alignment": {"color": "#111827", "linestyle": "-"},
    "external_sinkhorn_soft": {"color": "#2563eb", "linestyle": "-"},
    "external_sinkhorn_hard": {"color": "#2563eb", "linestyle": "--"},
}


def import_external_sinkhorn():
    """Import the vendored external Sinkhorn modules with a clear error path."""

    sinkhorn_root = project_root / "external" / "sinkhorn-rebasin"
    examples_root = sinkhorn_root / "examples"
    for path in (sinkhorn_root, examples_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        from models.vgg import VGG
        from rebasin import RebasinNet, matching
        from rebasin.loss import RndLoss
    except ImportError as exc:
        raise RuntimeError(
            "Unable to import external/sinkhorn-rebasin. "
            "The vendored baseline requires its Python dependencies, notably "
            "`torchviz` and `graphviz`, to be available in the current environment."
        ) from exc

    return VGG, RebasinNet, matching, RndLoss


def native_to_external_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Rename the local VGG16 state dict into the external VGG16 layout."""

    translated = OrderedDict()
    for native_prefix, external_prefix in CONV_KEY_MAP + LINEAR_KEY_MAP:
        translated[f"{external_prefix}.weight"] = state_dict[f"{native_prefix}.weight"].detach().cpu().clone()
        translated[f"{external_prefix}.bias"] = state_dict[f"{native_prefix}.bias"].detach().cpu().clone()
    return translated


def external_to_native_state_dict(state_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    """Rename the external VGG16 state dict back into the local checkpoint layout."""

    translated = OrderedDict()
    for native_prefix, external_prefix in CONV_KEY_MAP + LINEAR_KEY_MAP:
        translated[f"{native_prefix}.weight"] = state_dict[f"{external_prefix}.weight"].detach().cpu().clone()
        translated[f"{native_prefix}.bias"] = state_dict[f"{external_prefix}.bias"].detach().cpu().clone()
    return translated


def build_external_vgg(state_dict: Mapping[str, torch.Tensor], *, device: torch.device):
    """Instantiate the external VGG16 and load weights translated from local checkpoints."""

    VGG, _, _, _ = import_external_sinkhorn()
    model = VGG("VGG16", in_channels=3, out_features=10, h_in=32, w_in=32)
    model.load_state_dict(native_to_external_state_dict(state_dict))
    model.to(device)
    model.eval()
    return model


def build_calibration_loader(
    *,
    data_path: str,
    calibration_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Reuse the repo's eval-mode CIFAR loader for external baseline training."""

    loaders, _ = load_cifar10_eval_loaders(data_path=data_path, batch_size=batch_size, num_workers=num_workers)
    dataset = loaders["train"].dataset
    subset_size = min(calibration_size, len(dataset))
    subset = Subset(dataset, list(range(subset_size)))

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )


def cycle_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def evaluate_endpoint_metrics(
    state_dict: Mapping[str, torch.Tensor],
    loaders,
    *,
    device: torch.device,
    max_eval_batches: int | None = None,
) -> Dict[str, float]:
    """Evaluate one local-format state dict on train and test splits."""

    model = create_vgg16_model(num_classes=10, device=device)
    model.load_state_dict(OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items()))
    model.eval()

    def _metrics(loader) -> Dict[str, float]:
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(loader):
                if max_eval_batches is not None and batch_index >= max_eval_batches:
                    break
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits = model(inputs)
                total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
                total_correct += logits.argmax(dim=1).eq(targets).sum().item()
                total_examples += targets.size(0)
        if total_examples == 0:
            raise ValueError("Evaluation loader produced zero samples.")
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


def validate_state_translation(
    native_state: Mapping[str, torch.Tensor],
    external_model,
    *,
    device: torch.device,
) -> Dict[str, float | bool]:
    """Check that local and translated external models produce the same logits."""

    native_model = create_vgg16_model(num_classes=10, device=device)
    native_model.load_state_dict(OrderedDict((key, value.detach().cpu()) for key, value in native_state.items()))
    native_model.eval()
    external_model.eval()

    inputs = torch.randn(4, 3, 32, 32, device=device)
    with torch.no_grad():
        native_logits = native_model(inputs)
        external_logits = external_model(inputs)
    diff = torch.abs(native_logits - external_logits)
    return {
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "allclose": bool(torch.allclose(native_logits, external_logits, atol=1e-6, rtol=1e-6)),
    }


def plot_variant_curves(output_path: str, variant_results: Mapping[str, Dict[str, np.ndarray]]) -> None:
    """Write a compact train/test interpolation plot for the external baseline."""

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
    config_name="external_sinkhorn_rebasin_vgg16",
)
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))

    runtime_device = resolve_device(str(cfg.device))
    output_root = Path(to_absolute_path(cfg.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    model_a_checkpoint = to_absolute_path(cfg.model_a_checkpoint)
    model_b_checkpoint = to_absolute_path(cfg.model_b_checkpoint)
    data_path = to_absolute_path(cfg.data_path)

    state_a_native = load_checkpoint_state_dict(model_a_checkpoint)
    state_b_native = load_checkpoint_state_dict(model_b_checkpoint)

    model_a_external = build_external_vgg(state_a_native, device=runtime_device)
    model_b_external = build_external_vgg(state_b_native, device=runtime_device)

    translation_checks = {
        "model_a": validate_state_translation(state_a_native, model_a_external, device=runtime_device),
        "model_b": validate_state_translation(state_b_native, model_b_external, device=runtime_device),
    }

    _, RebasinNet, matching, RndLoss = import_external_sinkhorn()
    rebasin_net = RebasinNet(
        model_b_external,
        input_shape=(1, 3, 32, 32),
        l=float(cfg.sinkhorn_l),
        tau=float(cfg.tau),
        n_iter=int(cfg.sinkhorn_iters),
    )
    if bool(cfg.identity_init):
        rebasin_net.identity_init()
    rebasin_net.to(runtime_device)

    criterion = RndLoss(model_a_external, criterion=torch.nn.CrossEntropyLoss())
    criterion.to(runtime_device)
    optimizer = torch.optim.AdamW(rebasin_net.parameters(), lr=float(cfg.lr))

    calibration_loader = build_calibration_loader(
        data_path=data_path,
        calibration_size=int(cfg.calibration_size),
        batch_size=int(cfg.alignment_batch_size),
        num_workers=int(cfg.num_workers),
        seed=int(cfg.seed),
    )
    batch_iterator = cycle_loader(calibration_loader)

    history: list[Dict[str, Any]] = []
    print("=" * 80)
    print("EXTERNAL SINKHORN-REBASIN BASELINE")
    print("=" * 80)
    print(f"experiment_name: {cfg.experiment_name}")
    print(f"model_a_checkpoint: {model_a_checkpoint}")
    print(f"model_b_checkpoint: {model_b_checkpoint}")
    print(f"output_root: {output_root}")
    print(f"device: {runtime_device}")
    print("")

    for step in range(1, int(cfg.alignment_steps) + 1):
        inputs, targets = next(batch_iterator)
        inputs = inputs.to(runtime_device)
        targets = targets.to(runtime_device)

        rebasin_net.train()
        rebased_model = rebasin_net()
        loss = criterion(rebased_model, inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step_metrics = {
            "step": step,
            "loss": float(loss.detach().cpu().item()),
        }
        history.append(step_metrics)
        if step == 1 or step == int(cfg.alignment_steps) or (
            int(cfg.log_interval) > 0 and step % int(cfg.log_interval) == 0
        ):
            print(f"[external_sinkhorn] step={step:04d} loss={step_metrics['loss']:.4f}")

    rebasin_net.train()
    soft_model = deepcopy(rebasin_net())
    soft_model.eval()
    soft_state_native = external_to_native_state_dict(soft_model.state_dict())

    rebasin_net.eval()
    hard_model = deepcopy(rebasin_net())
    hard_model.eval()
    hard_state_native = external_to_native_state_dict(hard_model.state_dict())

    soft_checkpoint_path = str(output_root / "soft_aligned.pt")
    hard_checkpoint_path = str(output_root / "hard_aligned.pt")
    artifact_path = str(output_root / "alignment_artifacts.pt")
    history_path = str(output_root / "training_history.json")
    metadata_path = str(output_root / "metadata.json")

    permutation_matrices = [
        parameter.detach().cpu().clone() for parameter in rebasin_net.p if parameter is not None
    ]
    hard_permutations = [
        matching(parameter.detach().cpu().numpy()).to(torch.float32).cpu()
        for parameter in rebasin_net.p
        if parameter is not None
    ]
    artifact = {
        "raw_parameters": permutation_matrices,
        "hard_permutations": hard_permutations,
        "translation_checks": translation_checks,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    torch.save(artifact, artifact_path)

    save_checkpoint_with_state_dict(
        model_b_checkpoint,
        soft_checkpoint_path,
        soft_state_native,
        metadata={"method": "external_sinkhorn", "artifact_path": artifact_path, "metadata_path": metadata_path},
    )
    save_checkpoint_with_state_dict(
        model_b_checkpoint,
        hard_checkpoint_path,
        hard_state_native,
        metadata={"method": "external_sinkhorn", "artifact_path": artifact_path, "metadata_path": metadata_path},
    )
    save_json(history, history_path, indent=2)

    loaders, _ = load_cifar10_eval_loaders(
        data_path=data_path,
        batch_size=int(cfg.evaluation_batch_size),
        num_workers=int(cfg.num_workers),
    )
    endpoint_a = evaluate_endpoint_metrics(state_a_native, loaders, device=runtime_device, max_eval_batches=None)
    variants = {
        "no_alignment": state_b_native,
        "external_sinkhorn_soft": soft_state_native,
        "external_sinkhorn_hard": hard_state_native,
    }

    evaluation_dir = ensure_dir(output_root / "evaluation")
    variant_rows = []
    variant_results = {}
    for variant_key, variant_state in variants.items():
        interpolation = evaluate_linear_interpolation(
            state_a_native,
            variant_state,
            loaders,
            num_points=int(cfg.num_eval_points),
            device=runtime_device,
        )
        variant_dir = ensure_dir(Path(evaluation_dir) / variant_key)
        save_interpolation_results(str(Path(variant_dir) / "interpolation.npz"), interpolation)
        variant_results[variant_key] = interpolation

        endpoint_b = evaluate_endpoint_metrics(variant_state, loaders, device=runtime_device, max_eval_batches=None)
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
    print("EXTERNAL SINKHORN RUN COMPLETE")
    print("=" * 80)
    print(f"Soft checkpoint: {soft_checkpoint_path}")
    print(f"Hard checkpoint: {hard_checkpoint_path}")
    print(f"Artifacts: {artifact_path}")
    print(f"Evaluation summary: {Path(evaluation_dir) / 'comparison.json'}")
    print(f"Comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
