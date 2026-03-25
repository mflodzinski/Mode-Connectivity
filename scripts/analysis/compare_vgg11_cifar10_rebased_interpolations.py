from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
import torch
import torchvision
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".mplcache"))
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import build_model
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir


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


def load_model_from_checkpoint(*, checkpoint_path: Path, image_size: int, device: torch.device, VGGClass) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(
            f"Unsupported checkpoint payload at {checkpoint_path}; expected raw state_dict or dict with "
            "'model_state'/'state_dict'."
        )

    model = build_model(VGGClass, "VGG11", num_classes=10, image_size=image_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_cifar10_loaders(*, data_path: Path, batch_size: int, num_workers: int, dnn_data):
    transform_eval = dnn_data.Transforms.CIFAR10.VGG.test
    cifar_root = data_path / "cifar10"
    dataset_train = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=True,
        download=True,
        transform=transform_eval,
    )
    dataset_test = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=False,
        download=True,
        transform=transform_eval,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
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
    return train_loader, test_loader, str(transform_eval)


def evaluate_curve(
    *,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    num_eval_points: int,
    eval_loss_acc,
    lerp,
    device: torch.device,
) -> dict[str, list[float]]:
    lambdas = torch.linspace(0, 1, int(num_eval_points))
    losses: list[float] = []
    accuracies: list[float] = []
    errors: list[float] = []
    for lam in tqdm(lambdas.tolist(), leave=False):
        temporal_model = lerp(model_a, model_b, lam)
        loss_value, acc_value = eval_loss_acc(temporal_model, loader, torch.nn.CrossEntropyLoss(), device)
        acc_percent = float(acc_value) * 100.0
        losses.append(float(loss_value))
        accuracies.append(acc_percent)
        errors.append(100.0 - acc_percent)
    return {
        "lambdas": lambdas.tolist(),
        "losses": losses,
        "accuracies": accuracies,
        "errors": errors,
    }


def plot_two_curves(*, x: list[float], y_a: list[float], y_b: list[float], label_a: str, label_b: str, title: str, ylabel: str, output_path: Path) -> None:
    plt.figure()
    plt.plot(x, y_a, label=label_a)
    plt.plot(x, y_b, label=label_b)
    plt.xlabel("alpha")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare interpolation curves for VGG11/CIFAR10 rebased models with and without scale.")
    parser.add_argument(
        "--model-a-checkpoint",
        type=Path,
        default=Path("VGG11_cifar10_0.911.pth"),
    )
    parser.add_argument(
        "--model-b-checkpoint",
        type=Path,
        default=Path("VGG11_cifar10_0.9139.pth"),
    )
    parser.add_argument(
        "--rebased-no-scale-checkpoint",
        type=Path,
        default=Path("results/vgg11/cifar10/raw_pth_align_sweep/steps150_tau1p0_lr0p1_l1p0_lossmidpoint/rebased_model.pt"),
    )
    parser.add_argument(
        "--rebased-scale-checkpoint",
        type=Path,
        default=Path("results/vgg11/cifar10/raw_pth_align_sweep_scale/steps150_tau1p0_lr0p1_l1p0_lossmidpoint_lam0p005/rebased_model.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vgg11/cifar10/rebased_interpolation_comparison"),
    )
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-eval-points", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir.resolve())
    device = resolve_device(args.device)
    VGGClass, dnn_data, eval_loss_acc, lerp = import_components()
    train_loader, test_loader, eval_transform = build_cifar10_loaders(
        data_path=args.data_path.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dnn_data=dnn_data,
    )

    model_a = load_model_from_checkpoint(
        checkpoint_path=args.model_a_checkpoint.resolve(),
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    model_b = load_model_from_checkpoint(
        checkpoint_path=args.model_b_checkpoint.resolve(),
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    rebased_no_scale = load_model_from_checkpoint(
        checkpoint_path=args.rebased_no_scale_checkpoint.resolve(),
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    rebased_scale = load_model_from_checkpoint(
        checkpoint_path=args.rebased_scale_checkpoint.resolve(),
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )

    print("Computing no-scale train curve")
    train_no_scale = evaluate_curve(
        model_a=rebased_no_scale,
        model_b=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing no-scale test curve")
    test_no_scale = evaluate_curve(
        model_a=rebased_no_scale,
        model_b=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing scale train curve")
    train_scale = evaluate_curve(
        model_a=rebased_scale,
        model_b=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing scale test curve")
    test_scale = evaluate_curve(
        model_a=rebased_scale,
        model_b=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )

    plot_two_curves(
        x=test_no_scale["lambdas"],
        y_a=test_no_scale["losses"],
        y_b=test_scale["losses"],
        label_a="No scale",
        label_b="With scale",
        title="Test Loss Along Interpolation",
        ylabel="Cross-entropy",
        output_path=output_dir / "compare_test_loss.png",
    )
    plot_two_curves(
        x=test_no_scale["lambdas"],
        y_a=test_no_scale["accuracies"],
        y_b=test_scale["accuracies"],
        label_a="No scale",
        label_b="With scale",
        title="Test Accuracy Along Interpolation",
        ylabel="Accuracy (%)",
        output_path=output_dir / "compare_test_accuracy.png",
    )
    plot_two_curves(
        x=test_no_scale["lambdas"],
        y_a=test_no_scale["errors"],
        y_b=test_scale["errors"],
        label_a="No scale",
        label_b="With scale",
        title="Test Error Along Interpolation",
        ylabel="Error (%)",
        output_path=output_dir / "compare_test_error.png",
    )

    payload = {
        "model_a_checkpoint": str(args.model_a_checkpoint.resolve()),
        "model_b_checkpoint": str(args.model_b_checkpoint.resolve()),
        "rebased_no_scale_checkpoint": str(args.rebased_no_scale_checkpoint.resolve()),
        "rebased_scale_checkpoint": str(args.rebased_scale_checkpoint.resolve()),
        "data_path": str(args.data_path.resolve()),
        "eval_transform": eval_transform,
        "num_eval_points": int(args.num_eval_points),
        "device": str(device),
        "curves": {
            "train_no_scale": train_no_scale,
            "test_no_scale": test_no_scale,
            "train_scale": train_scale,
            "test_scale": test_scale,
        },
    }
    with open(output_dir / "curves.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
