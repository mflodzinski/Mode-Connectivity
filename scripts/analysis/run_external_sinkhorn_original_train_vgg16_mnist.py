"""Train two upstream sinkhorn-rebasin VGG16 MNIST models with early stopping."""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".mplcache"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from src.utils import set_global_seed
from scripts.analysis.run_external_sinkhorn_original_train_align_vgg16_mnist import (
    build_mnist_loaders,
    import_original_components,
)
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir, save_json


def format_metric(value: float) -> str:
    return f"{value:.6f}" if abs(value) < 1e-3 else f"{value:.4f}"


def evaluate_loss_acc(model, loader, criterion, device: torch.device) -> tuple[float, float]:
    model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    param_dtype = next(iter(model.parameters())).dtype
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device=device, dtype=param_dtype)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * inputs.shape[0]
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_examples += inputs.shape[0]

    if total_examples == 0:
        raise ValueError("Loader produced zero examples during evaluation.")
    return total_loss / total_examples, total_correct / total_examples


def train_one_model_with_early_stopping(
    *,
    model_name: str,
    seed: int,
    VGG,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    cfg: DictConfig,
) -> tuple[torch.nn.Module, dict[str, Any], list[dict[str, float | int]]]:
    """Train one upstream VGG16 model with patience-based early stopping."""

    set_global_seed(seed)
    model = VGG(
        "VGG16",
        in_channels=1,
        out_features=10,
        h_in=int(cfg.image_size),
        w_in=int(cfg.image_size),
    )
    model.to(device)

    optimizer_name = str(cfg.optimizer_name).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.train_lr), weight_decay=float(cfg.weight_decay))
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.train_lr), weight_decay=float(cfg.weight_decay))
    else:
        raise ValueError(f"Unsupported optimizer_name={cfg.optimizer_name!r}. Expected 'adamw' or 'adam'.")

    criterion = torch.nn.CrossEntropyLoss()
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(cfg.train_epochs) + 1):
        model.train()
        param_dtype = next(iter(model.parameters())).dtype
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_examples = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device=device, dtype=param_dtype)
            targets = targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * inputs.shape[0]
            total_train_correct += (logits.argmax(dim=1) == targets).sum().item()
            total_train_examples += inputs.shape[0]

        train_loss = total_train_loss / total_train_examples
        train_acc = total_train_correct / total_train_examples
        val_loss, val_acc = evaluate_loss_acc(model, val_loader, criterion, device)

        improved = val_loss < (best_val_loss - float(cfg.min_delta))
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "best_val_loss_so_far": float(best_val_loss),
                "patience_counter": patience_counter,
            }
        )

        should_log = epoch == 1 or epoch == int(cfg.train_epochs) or (
            int(cfg.log_interval) > 0 and epoch % int(cfg.log_interval) == 0
        )
        if should_log:
            print(
                f"[{model_name}] epoch={epoch:03d} "
                f"train_loss={format_metric(train_loss)} train_acc={train_acc * 100.0:.2f} "
                f"val_loss={format_metric(val_loss)} val_acc={val_acc * 100.0:.2f} "
                f"best_val_loss={format_metric(best_val_loss)} patience={patience_counter}/{int(cfg.early_stopping_patience)}"
            )

        if patience_counter >= int(cfg.early_stopping_patience):
            print(f"[{model_name}] early stopping at epoch {epoch:03d}; best epoch was {best_epoch:03d}")
            break

    model.load_state_dict(best_state)
    model.eval()
    test_loss, test_acc = evaluate_loss_acc(model, test_loader, criterion, device)
    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }
    return model, metrics, history


def save_model_checkpoint(path: Path, model: torch.nn.Module, metadata: dict[str, Any]) -> None:
    torch.save(
        {
            "model_state": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
            "metadata": metadata,
        },
        path,
    )


def run_original_train_vgg16_mnist(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))

    (
        VGG,
        _RebasinNet,
        _matching,
        MNistDataset,
        SmallMNistDataset,
        _MidLoss,
        _RndLoss,
        _DistL1Loss,
        _DistL2Loss,
        _DistCosineLoss,
        _train,
        _eval_loss_acc,
        _lerp,
    ) = import_original_components()

    runtime_device = resolve_device(str(cfg.device))
    output_root = ensure_dir(Path(to_absolute_path(str(cfg.output_root))))
    endpoints_dir = ensure_dir(output_root / "endpoints")

    train_loader, val_loader, test_loader = build_mnist_loaders(
        cfg,
        MNistDataset=MNistDataset,
        SmallMNistDataset=SmallMNistDataset,
    )

    print("=" * 80)
    print("ORIGINAL SINKHORN VGG16 MNIST TRAINING")
    print("=" * 80)
    print(f"output_root: {output_root}")
    print(f"device: {runtime_device}")
    print(f"use_small_dataset: {bool(cfg.use_small_dataset)}")
    print(f"image_size: {int(cfg.image_size)}")
    print(f"optimizer: {cfg.optimizer_name}")
    print(f"train_lr: {cfg.train_lr}")
    print(f"early_stopping_patience: {cfg.early_stopping_patience}")
    print(f"min_delta: {cfg.min_delta}")
    print("")

    model_a, model_a_metrics, model_a_history = train_one_model_with_early_stopping(
        model_name="model_a",
        seed=int(cfg.model_a_seed),
        VGG=VGG,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=runtime_device,
        cfg=cfg,
    )
    print(
        f"Model A best_epoch={model_a_metrics['best_epoch']} "
        f"best_val_loss={format_metric(model_a_metrics['best_val_loss'])} "
        f"test_loss={format_metric(model_a_metrics['test_loss'])} "
        f"test_acc={model_a_metrics['test_acc'] * 100.0:.2f}"
    )

    model_b, model_b_metrics, model_b_history = train_one_model_with_early_stopping(
        model_name="model_b",
        seed=int(cfg.model_b_seed),
        VGG=VGG,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=runtime_device,
        cfg=cfg,
    )
    print(
        f"Model B best_epoch={model_b_metrics['best_epoch']} "
        f"best_val_loss={format_metric(model_b_metrics['best_val_loss'])} "
        f"test_loss={format_metric(model_b_metrics['test_loss'])} "
        f"test_acc={model_b_metrics['test_acc'] * 100.0:.2f}"
    )

    model_a_path = endpoints_dir / "model_a.pt"
    model_b_path = endpoints_dir / "model_b.pt"
    save_model_checkpoint(model_a_path, model_a, model_a_metrics)
    save_model_checkpoint(model_b_path, model_b, model_b_metrics)
    save_json(model_a_history, output_root / "model_a_history.json", indent=2)
    save_json(model_b_history, output_root / "model_b_history.json", indent=2)

    metadata = {
        "experiment_name": str(cfg.experiment_name),
        "output_root": str(output_root),
        "model_a_path": str(model_a_path),
        "model_b_path": str(model_b_path),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "model_a_metrics": model_a_metrics,
        "model_b_metrics": model_b_metrics,
    }
    save_json(metadata, output_root / "metadata.json", indent=2)

    print("")
    print("=" * 80)
    print("ORIGINAL SINKHORN VGG16 MNIST TRAINING COMPLETE")
    print("=" * 80)
    print(f"Model A checkpoint: {model_a_path}")
    print(f"Model B checkpoint: {model_b_path}")
    print(f"Metadata: {output_root / 'metadata.json'}")

    return metadata


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_train_vgg16_mnist",
)
def main(cfg: DictConfig) -> None:
    run_original_train_vgg16_mnist(cfg)


if __name__ == "__main__":
    main()
