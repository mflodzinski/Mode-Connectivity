"""Evaluate saved original-sinkhorn VGG11 MNIST endpoints on train/val/test splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
import torchvision
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import (
    build_model,
    evaluate_model,
    import_original_mnist_components,
)
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir, save_json
from src.utils import set_global_seed


def load_endpoint_model(
    *,
    checkpoint_path: Path,
    vgg_name: str,
    image_size: int,
    device: torch.device,
    VGGClass,
) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a 'model_state' entry.")
    model = build_model(VGGClass, vgg_name=vgg_name, num_classes=10, image_size=image_size)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_vgg11_mnist_endpoint_eval",
)
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    output_root = ensure_dir(Path(to_absolute_path(str(cfg.output_root))))

    VGGClass, _, _, dnn_data, _, _, _ = import_original_mnist_components()

    vgg_name = str(cfg.vgg_name)
    image_size = int(cfg.image_size)
    if image_size != 32:
        raise ValueError(f"This {vgg_name} MNIST VGG pipeline requires image_size=32.")

    transform_train = dnn_data.Transforms.MNIST.VGG.train
    transform_test = dnn_data.Transforms.MNIST.VGG.test
    mnist_root = Path(to_absolute_path(str(cfg.data_path))) / "mnist"

    dataset_train_source = torchvision.datasets.MNIST(
        root=str(mnist_root),
        train=True,
        download=True,
        transform=transform_train,
    )
    dataset_val_source = torchvision.datasets.MNIST(
        root=str(mnist_root),
        train=True,
        download=True,
        transform=transform_test,
    )
    dataset_test_source = torchvision.datasets.MNIST(
        root=str(mnist_root),
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

    batch_size = int(cfg.batch_size)
    num_workers = int(cfg.num_workers)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    criterion = torch.nn.CrossEntropyLoss()

    model_a_path = Path(to_absolute_path(str(cfg.model_a_checkpoint)))
    model_b_path = Path(to_absolute_path(str(cfg.model_b_checkpoint)))
    model_a = load_endpoint_model(
        checkpoint_path=model_a_path,
        vgg_name=vgg_name,
        image_size=image_size,
        device=device,
        VGGClass=VGGClass,
    )
    model_b = load_endpoint_model(
        checkpoint_path=model_b_path,
        vgg_name=vgg_name,
        image_size=image_size,
        device=device,
        VGGClass=VGGClass,
    )

    print("=" * 80)
    print(f"ORIGINAL SINKHORN {vgg_name} MNIST ENDPOINT EVAL")
    print("=" * 80)
    print(f"output_root: {output_root}")
    print(f"model_a_checkpoint: {model_a_path}")
    print(f"model_b_checkpoint: {model_b_path}")
    print(f"device: {device}")
    print(f"image_size: {image_size}")
    print(f"train_transform: {transform_train}")
    print(f"test_transform: {transform_test}")
    print(f"dataset_split_sizes: train={train_size}, val={val_size}, test={len(dataset_test)}")
    print(f"batch_size: {batch_size}")
    print("")

    results: dict[str, Any] = {
        "experiment_name": str(cfg.experiment_name),
        "output_root": str(output_root),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "splits": {
            "train_size": train_size,
            "val_size": val_size,
            "test_size": len(dataset_test),
        },
        "models": {},
    }

    for model_name, model in (("model_a", model_a), ("model_b", model_b)):
        train_loss, train_acc = evaluate_model(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
        results["models"][model_name] = {
            "train_loss": float(train_loss),
            "train_acc": float(train_acc) * 100.0,
            "val_loss": float(val_loss),
            "val_acc": float(val_acc) * 100.0,
            "test_loss": float(test_loss),
            "test_acc": float(test_acc) * 100.0,
        }
        print(
            f"{model_name}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc * 100.0:.2f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc * 100.0:.2f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc * 100.0:.2f}"
        )

    save_json(results, output_root / "endpoint_metrics.json", indent=2)
    print("")
    print(f"Metrics written to: {output_root / 'endpoint_metrics.json'}")


if __name__ == "__main__":
    main()
