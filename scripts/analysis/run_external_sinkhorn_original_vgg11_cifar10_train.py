"""Train sinkhorn-rebasin VGG11 endpoints on CIFAR10."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import hydra
import torch
import torchvision
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import (
    build_model,
    evaluate_model,
    import_original_mnist_components,
    save_model_checkpoint,
    train_model,
)
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir, save_json
from src.utils import set_global_seed


def build_cifar10_loaders(cfg: DictConfig, dnn_data):
    transform_train = dnn_data.Transforms.CIFAR10.VGG.train
    transform_test = dnn_data.Transforms.CIFAR10.VGG.test
    cifar_root = Path(to_absolute_path(str(cfg.data_path))) / "cifar10"

    dataset_train_source = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=True,
        download=True,
        transform=transform_train,
    )
    dataset_val_source = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=True,
        download=True,
        transform=transform_test,
    )
    dataset_test_source = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=False,
        download=True,
        transform=transform_test,
    )

    train_total_size = len(dataset_train_source)
    val_fraction = float(cfg.val_fraction)
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")
    val_size = int(train_total_size * val_fraction)
    train_size = train_total_size - val_size
    indices = torch.randperm(train_total_size, generator=torch.Generator().manual_seed(int(cfg.split_seed)))
    train_indices = indices[:train_size].tolist()
    val_indices = indices[train_size:].tolist()

    dataset_train = torch.utils.data.Subset(dataset_train_source, train_indices)
    dataset_val = torch.utils.data.Subset(dataset_val_source, val_indices)
    dataset_test = dataset_test_source

    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, val_loader, test_loader, train_size, val_size, len(dataset_test), transform_train, transform_test


def run_original_vgg11_cifar10_train(cfg: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(dict(cfg))

    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    output_root = ensure_dir(Path(to_absolute_path(str(cfg.output_root))))

    VGGClass, _, _, dnn_data, _, _, _ = import_original_mnist_components()
    train_loader, val_loader, test_loader, train_size, val_size, test_size, transform_train, transform_test = build_cifar10_loaders(cfg, dnn_data)

    print("=" * 80)
    print("ORIGINAL SINKHORN VGG11 CIFAR10 TRAINING")
    print("=" * 80)
    print(f"output_root: {output_root}")
    print(f"device: {device}")
    print(f"train_transform: {transform_train}")
    print(f"test_transform: {transform_test}")
    print(f"dataset_split_sizes: train={train_size}, val={val_size}, test={test_size}")
    print(f"batch_size: {int(cfg.batch_size)}")
    print(f"train_epochs: {int(cfg.train_epochs)}")
    print(f"train_lr: {float(cfg.train_lr)}")
    print(f"momentum: {float(cfg.momentum)}")
    print(f"weight_decay: {float(cfg.weight_decay)}")
    print("")

    model_a = build_model(VGGClass, "VGG11", num_classes=10, image_size=int(cfg.image_size))
    print("Training network A")
    model_a = train_model(
        model_a,
        train_loader,
        val_loader,
        device,
        int(cfg.train_epochs),
        base_lr=float(cfg.train_lr),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
        early_stopping_patience=int(cfg.early_stopping_patience),
        min_delta=float(cfg.min_delta),
    )
    loss_a, acc_a = evaluate_model(model_a, test_loader, torch.nn.CrossEntropyLoss(), device)
    print("Model A: test loss {:1.3f}, test accuracy {:1.3f}".format(loss_a, acc_a))
    model_a.eval()

    model_b = build_model(VGGClass, "VGG11", num_classes=10, image_size=int(cfg.image_size))
    print("\nTraining network B")
    model_b = train_model(
        model_b,
        train_loader,
        val_loader,
        device,
        int(cfg.train_epochs),
        base_lr=float(cfg.train_lr),
        momentum=float(cfg.momentum),
        weight_decay=float(cfg.weight_decay),
        early_stopping_patience=int(cfg.early_stopping_patience),
        min_delta=float(cfg.min_delta),
    )
    loss_b, acc_b = evaluate_model(model_b, test_loader, torch.nn.CrossEntropyLoss(), device)
    print("Model B: test loss {:1.3f}, test accuracy {:1.3f}".format(loss_b, acc_b))
    model_b.eval()

    save_model_checkpoint(
        output_root / "model_a.pt",
        model_a,
        {"test_loss": float(loss_a), "test_acc": float(acc_a), "architecture": "VGG11", "dataset": "CIFAR10"},
    )
    save_model_checkpoint(
        output_root / "model_b.pt",
        model_b,
        {"test_loss": float(loss_b), "test_acc": float(acc_b), "architecture": "VGG11", "dataset": "CIFAR10"},
    )

    metadata = {
        "experiment_name": str(cfg.experiment_name),
        "output_root": str(output_root),
        "model_a_test_loss": float(loss_a),
        "model_a_test_acc": float(acc_a) * 100.0,
        "model_b_test_loss": float(loss_b),
        "model_b_test_acc": float(acc_b) * 100.0,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    save_json(metadata, output_root / "metadata.json", indent=2)
    print("")
    print("=" * 80)
    print("ORIGINAL SINKHORN VGG11 CIFAR10 TRAINING COMPLETE")
    print("=" * 80)
    print(f"Artifacts written under: {output_root}")
    return metadata


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_vgg11_cifar10_train",
)
def main(cfg: DictConfig) -> None:
    run_original_vgg11_cifar10_train(cfg)


if __name__ == "__main__":
    main()
