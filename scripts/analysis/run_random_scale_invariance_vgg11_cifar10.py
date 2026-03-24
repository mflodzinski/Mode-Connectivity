"""Apply random compensated scales to VGG11/CIFAR10 and evaluate interpolation."""

from __future__ import annotations

import math
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import matplotlib
import torch
import torchvision
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import (
    build_model,
)
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir, save_json
from src.utils import set_global_seed


def import_components():
    sinkhorn_root = project_root / "external" / "sinkhorn-rebasin"
    examples_root = sinkhorn_root / "examples"
    dnn_root = project_root / "external" / "dnn-mode-connectivity"
    for path in (str(examples_root), str(sinkhorn_root), str(dnn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    sinkhorn_vgg_path = examples_root / "models" / "vgg.py"
    spec = importlib.util.spec_from_file_location("_sinkhorn_rebasin_examples_vgg", sinkhorn_vgg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load sinkhorn-rebasin VGG definition from {sinkhorn_vgg_path}.")
    vgg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vgg_module)
    VGG = vgg_module.VGG

    import data as dnn_data
    from utils import eval_loss_acc, lerp

    return VGG, dnn_data, eval_loss_acc, lerp


def build_cifar10_eval_loaders(cfg: DictConfig, dnn_data):
    transform_eval = dnn_data.Transforms.CIFAR10.VGG.test
    cifar_root = Path(to_absolute_path(str(cfg.data_path))) / "cifar10"
    train_dataset = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=True,
        download=True,
        transform=transform_eval,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=False,
        download=True,
        transform=transform_eval,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=int(cfg.batch_size),
        shuffle=False,
        num_workers=int(cfg.num_workers),
    )
    return train_loader, test_loader, str(transform_eval)


def load_model_from_checkpoint(model_path: Path, VGGClass, *, image_size: int, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = build_model(VGGClass, "VGG11", num_classes=10, image_size=image_size)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint payload at {model_path}; expected a raw state_dict or dict with "
            "'model_state'/'state_dict'."
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def affine_layers(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    layers: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            layers.append((name, module))
    return layers


def sample_positive_scales(num_units: int, *, min_scale: float, max_scale: float, sampling: str, generator: torch.Generator) -> torch.Tensor:
    if sampling == "uniform":
        return torch.empty(num_units).uniform_(float(min_scale), float(max_scale), generator=generator)
    if sampling == "log_uniform":
        low = math.log(float(min_scale))
        high = math.log(float(max_scale))
        return torch.empty(num_units).uniform_(low, high, generator=generator).exp()
    raise ValueError(f"Unsupported sampling={sampling!r}. Expected 'uniform' or 'log_uniform'.")


def apply_output_scale(module: torch.nn.Module, scale: torch.Tensor) -> None:
    with torch.no_grad():
        if isinstance(module, torch.nn.Conv2d):
            module.weight.mul_(scale.view(-1, 1, 1, 1).to(device=module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None:
                module.bias.mul_(scale.to(device=module.bias.device, dtype=module.bias.dtype))
            return
        if isinstance(module, torch.nn.Linear):
            module.weight.mul_(scale.view(-1, 1).to(device=module.weight.device, dtype=module.weight.dtype))
            if module.bias is not None:
                module.bias.mul_(scale.to(device=module.bias.device, dtype=module.bias.dtype))
            return
    raise TypeError(f"Unsupported module type for output scaling: {type(module)}")


def apply_input_inverse_scale(module: torch.nn.Module, inv_scale: torch.Tensor) -> None:
    with torch.no_grad():
        if isinstance(module, torch.nn.Conv2d):
            module.weight.mul_(inv_scale.view(1, -1, 1, 1).to(device=module.weight.device, dtype=module.weight.dtype))
            return
        if isinstance(module, torch.nn.Linear):
            expanded_inv_scale = inv_scale
            if module.in_features != inv_scale.numel():
                if module.in_features % inv_scale.numel() != 0:
                    raise ValueError(
                        f"Cannot broadcast inverse scale of length {inv_scale.numel()} to linear layer with "
                        f"in_features={module.in_features}."
                    )
                repeat_factor = module.in_features // inv_scale.numel()
                expanded_inv_scale = inv_scale.repeat_interleave(repeat_factor)
            module.weight.mul_(expanded_inv_scale.view(1, -1).to(device=module.weight.device, dtype=module.weight.dtype))
            return
    raise TypeError(f"Unsupported module type for input inverse scaling: {type(module)}")


def apply_random_compensated_scales(
    model: torch.nn.Module,
    *,
    min_scale: float,
    max_scale: float,
    sampling: str,
    seed: int,
) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    scaled_model = deepcopy(model)
    layers = affine_layers(scaled_model)
    generator = torch.Generator().manual_seed(int(seed))
    scale_records: list[dict[str, Any]] = []

    for layer_index, (layer_name, current_layer) in enumerate(layers[:-1]):
        next_layer_name, next_layer = layers[layer_index + 1]
        num_units = current_layer.out_channels if isinstance(current_layer, torch.nn.Conv2d) else current_layer.out_features
        scale = sample_positive_scales(
            num_units,
            min_scale=min_scale,
            max_scale=max_scale,
            sampling=sampling,
            generator=generator,
        )
        inv_scale = torch.reciprocal(scale)
        apply_output_scale(current_layer, scale)
        apply_input_inverse_scale(next_layer, inv_scale)
        scale_records.append(
            {
                "layer_index": int(layer_index),
                "layer_name": layer_name,
                "next_layer_name": next_layer_name,
                "num_units": int(num_units),
                "scale_min": float(scale.min().item()),
                "scale_mean": float(scale.mean().item()),
                "scale_max": float(scale.max().item()),
                "scale_std": float(scale.std(unbiased=False).item()),
            }
        )

    scaled_model.eval()
    return scaled_model, scale_records


@torch.no_grad()
def compute_equivalence_metrics(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    *,
    device: torch.device,
) -> dict[str, float]:
    total_examples = 0
    total_agreement = 0
    total_abs = 0.0
    total_sq = 0.0
    max_abs = 0.0
    for x, _ in loader:
        x = x.to(device)
        logits_a = model_a(x)
        logits_b = model_b(x)
        diff = logits_a - logits_b
        total_examples += x.shape[0]
        total_agreement += int((logits_a.argmax(dim=1) == logits_b.argmax(dim=1)).sum().item())
        total_abs += float(diff.abs().sum().item())
        total_sq += float(diff.pow(2).sum().item())
        max_abs = max(max_abs, float(diff.abs().max().item()))

    num_logits = total_examples * 10
    return {
        "prediction_agreement": 100.0 * total_agreement / total_examples,
        "mean_abs_logit_diff": total_abs / num_logits,
        "rmse_logit_diff": math.sqrt(total_sq / num_logits),
        "max_abs_logit_diff": max_abs,
    }


def evaluate_interpolation_curve(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    *,
    num_eval_points: int,
    lerp,
    eval_loss_acc,
    device: torch.device,
) -> dict[str, list[float]]:
    lambdas = torch.linspace(0, 1, int(num_eval_points))
    losses: list[float] = []
    accs: list[float] = []
    for lam in tqdm(lambdas.tolist(), leave=False):
        temporal_model = lerp(model_a, model_b, lam)
        loss_value, acc_value = eval_loss_acc(temporal_model, loader, torch.nn.CrossEntropyLoss(), device)
        losses.append(float(loss_value))
        accs.append(float(acc_value) * 100.0)
    return {"lambdas": lambdas.tolist(), "losses": losses, "accs": accs}


def run_experiment(cfg: DictConfig) -> None:
    set_global_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    output_root = ensure_dir(Path(to_absolute_path(str(cfg.output_root))))

    VGGClass, dnn_data, eval_loss_acc, lerp = import_components()
    train_loader, test_loader, eval_transform = build_cifar10_eval_loaders(cfg, dnn_data)
    model_path = Path(to_absolute_path(str(cfg.model_checkpoint)))
    model = load_model_from_checkpoint(model_path, VGGClass, image_size=int(cfg.image_size), device=device)
    scaled_model, scale_records = apply_random_compensated_scales(
        model,
        min_scale=float(cfg.scale_min),
        max_scale=float(cfg.scale_max),
        sampling=str(cfg.scale_sampling),
        seed=int(cfg.scale_seed),
    )
    equivalence_metrics = compute_equivalence_metrics(model, scaled_model, test_loader, device=device)

    train_original_loss, train_original_acc = eval_loss_acc(model, train_loader, torch.nn.CrossEntropyLoss(), device)
    test_original_loss, test_original_acc = eval_loss_acc(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    train_scaled_loss, train_scaled_acc = eval_loss_acc(scaled_model, train_loader, torch.nn.CrossEntropyLoss(), device)
    test_scaled_loss, test_scaled_acc = eval_loss_acc(scaled_model, test_loader, torch.nn.CrossEntropyLoss(), device)

    print("=" * 80)
    print("RANDOM SCALE INVARIANCE VGG11 CIFAR10")
    print("=" * 80)
    print(f"model_checkpoint: {model_path}")
    print(f"output_root: {output_root}")
    print(f"scale_range: [{float(cfg.scale_min)}, {float(cfg.scale_max)}]")
    print(f"scale_sampling: {cfg.scale_sampling}")
    print(f"scale_seed: {int(cfg.scale_seed)}")
    print(f"eval_transform: {eval_transform}")
    print(f"device: {device}")
    print("")
    print(
        "original train loss {:.4f}, train acc {:.2f}, test loss {:.4f}, test acc {:.2f}".format(
            float(train_original_loss),
            float(train_original_acc) * 100.0,
            float(test_original_loss),
            float(test_original_acc) * 100.0,
        )
    )
    print(
        "scaled   train loss {:.4f}, train acc {:.2f}, test loss {:.4f}, test acc {:.2f}".format(
            float(train_scaled_loss),
            float(train_scaled_acc) * 100.0,
            float(test_scaled_loss),
            float(test_scaled_acc) * 100.0,
        )
    )
    print(
        "equivalence: agreement {:.2f}%, mean_abs_logit_diff {:.6f}, rmse {:.6f}, max_abs {:.6f}".format(
            equivalence_metrics["prediction_agreement"],
            equivalence_metrics["mean_abs_logit_diff"],
            equivalence_metrics["rmse_logit_diff"],
            equivalence_metrics["max_abs_logit_diff"],
        )
    )

    print("")
    print("Evaluating interpolation on train set")
    train_curve = evaluate_interpolation_curve(
        model,
        scaled_model,
        train_loader,
        num_eval_points=int(cfg.num_eval_points),
        lerp=lerp,
        eval_loss_acc=eval_loss_acc,
        device=device,
    )
    print("Evaluating interpolation on test set")
    test_curve = evaluate_interpolation_curve(
        model,
        scaled_model,
        test_loader,
        num_eval_points=int(cfg.num_eval_points),
        lerp=lerp,
        eval_loss_acc=eval_loss_acc,
        device=device,
    )

    plt.figure()
    plt.plot(train_curve["lambdas"], train_curve["losses"], label="Train")
    plt.plot(test_curve["lambdas"], test_curve["losses"], label="Test")
    plt.title("Loss Along Interpolation")
    plt.xlabel("alpha")
    plt.ylabel("Cross-entropy")
    plt.legend()
    plt.savefig(output_root / "scale_interp_loss.png", dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_curve["lambdas"], train_curve["accs"], label="Train")
    plt.plot(test_curve["lambdas"], test_curve["accs"], label="Test")
    plt.title("Accuracy Along Interpolation")
    plt.xlabel("alpha")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig(output_root / "scale_interp_accuracy.png", dpi=200, bbox_inches="tight")
    plt.close()

    scale_values = [record["scale_min"] for record in scale_records] + [record["scale_max"] for record in scale_records]
    metadata = {
        "model_checkpoint": str(model_path),
        "output_root": str(output_root),
        "scale_min": float(cfg.scale_min),
        "scale_max": float(cfg.scale_max),
        "scale_sampling": str(cfg.scale_sampling),
        "scale_seed": int(cfg.scale_seed),
        "num_eval_points": int(cfg.num_eval_points),
        "batch_size": int(cfg.batch_size),
        "eval_transform": eval_transform,
        "original_metrics": {
            "train_loss": float(train_original_loss),
            "train_acc": float(train_original_acc) * 100.0,
            "test_loss": float(test_original_loss),
            "test_acc": float(test_original_acc) * 100.0,
        },
        "scaled_metrics": {
            "train_loss": float(train_scaled_loss),
            "train_acc": float(train_scaled_acc) * 100.0,
            "test_loss": float(test_scaled_loss),
            "test_acc": float(test_scaled_acc) * 100.0,
        },
        "equivalence_metrics": equivalence_metrics,
        "scale_stats_global": {
            "scale_min": float(min(record["scale_min"] for record in scale_records)),
            "scale_mean": float(sum(record["scale_mean"] for record in scale_records) / len(scale_records)),
            "scale_max": float(max(record["scale_max"] for record in scale_records)),
        },
        "layer_scale_stats": scale_records,
    }
    save_json(metadata, output_root / "metadata.json", indent=2)
    save_json(
        {
            "train_curve": train_curve,
            "test_curve": test_curve,
        },
        output_root / "interpolation_curves.json",
        indent=2,
    )


@hydra.main(
    version_base=None,
    config_path="../../configs/analysis",
    config_name="random_scale_invariance_vgg11_cifar10",
)
def main(cfg: DictConfig) -> None:
    run_experiment(cfg)


if __name__ == "__main__":
    main()
