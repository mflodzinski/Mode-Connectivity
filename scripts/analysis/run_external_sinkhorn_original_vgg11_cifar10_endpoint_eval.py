"""Evaluate saved original-sinkhorn VGG11 CIFAR10 endpoints on train/val/test splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import (
    build_model,
    evaluate_model,
    import_original_mnist_components,
)
from scripts.analysis.run_external_sinkhorn_original_vgg11_cifar10_train import build_cifar10_loaders
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir, save_json
from src.utils import set_global_seed


def load_endpoint_model(
    *,
    checkpoint_path: Path,
    image_size: int,
    device: torch.device,
    VGGClass,
) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError(f"Checkpoint at {checkpoint_path} does not contain a 'model_state' entry.")
    model = build_model(VGGClass, "VGG11", num_classes=10, image_size=image_size)
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="external_sinkhorn_original_vgg11_cifar10_endpoint_eval",
)
def main(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    output_root = ensure_dir(Path(to_absolute_path(str(cfg.output_root))))

    VGGClass, _, _, dnn_data, _, _, _ = import_original_mnist_components()
    train_loader, val_loader, test_loader, train_size, val_size, test_size, transform_train, transform_test = build_cifar10_loaders(cfg, dnn_data)

    model_a_path = Path(to_absolute_path(str(cfg.model_a_checkpoint)))
    model_b_path = Path(to_absolute_path(str(cfg.model_b_checkpoint)))
    model_a = load_endpoint_model(
        checkpoint_path=model_a_path,
        image_size=int(cfg.image_size),
        device=device,
        VGGClass=VGGClass,
    )
    model_b = load_endpoint_model(
        checkpoint_path=model_b_path,
        image_size=int(cfg.image_size),
        device=device,
        VGGClass=VGGClass,
    )

    criterion = torch.nn.CrossEntropyLoss()

    print("=" * 80)
    print("ORIGINAL SINKHORN VGG11 CIFAR10 ENDPOINT EVAL")
    print("=" * 80)
    print(f"output_root: {output_root}")
    print(f"model_a_checkpoint: {model_a_path}")
    print(f"model_b_checkpoint: {model_b_path}")
    print(f"device: {device}")
    print(f"train_transform: {transform_train}")
    print(f"test_transform: {transform_test}")
    print(f"dataset_split_sizes: train={train_size}, val={val_size}, test={test_size}")
    print(f"batch_size: {int(cfg.batch_size)}")
    print("")

    results: dict[str, Any] = {
        "experiment_name": str(cfg.experiment_name),
        "output_root": str(output_root),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "splits": {
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
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
