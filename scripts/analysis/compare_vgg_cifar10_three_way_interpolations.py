from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
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
from scripts.analysis.sinkhorn_experiment_utils import normalize_state_dict_keys
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


def load_model_from_checkpoint(*, checkpoint_path: Path, vgg_name: str, image_size: int, device: torch.device, VGGClass) -> torch.nn.Module:
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

    model = build_model(VGGClass, vgg_name, num_classes=10, image_size=image_size)
    model.load_state_dict(normalize_state_dict_keys(state_dict))
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


def plot_three_curves(
    *,
    x: list[float],
    y_naive: list[float],
    y_perm: list[float],
    y_scale: list[float],
    title: str,
    ylabel: str,
    output_path: Path,
    show_legend: bool,
) -> None:
    plt.figure()
    plt.plot(x, y_naive, label="No Alignment", color="tab:gray", linewidth=2.0)
    plt.plot(x, y_perm, label="Sinkhorn Permutation Only (From Scratch)", color="tab:orange", linewidth=2.0)
    plt.plot(x, y_scale, label="Sinkhorn Permutation + Scale (From Scratch)", color="tab:purple", linewidth=2.0)
    plt.xlabel("t (interpolation parameter)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
    if show_legend:
        plt.legend()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_curve_arrays(output_dir: Path, curves: dict[str, dict[str, list[float]]]) -> None:
    arrays_dir = ensure_dir(output_dir / "arrays")
    perm_bundle = {
        "lambdas": np.asarray(curves["test_naive"]["lambdas"], dtype=np.float64),
        "train_naive_losses": np.asarray(curves["train_naive"]["losses"], dtype=np.float64),
        "train_naive_accuracies": np.asarray(curves["train_naive"]["accuracies"], dtype=np.float64),
        "train_naive_errors": np.asarray(curves["train_naive"]["errors"], dtype=np.float64),
        "test_naive_losses": np.asarray(curves["test_naive"]["losses"], dtype=np.float64),
        "test_naive_accuracies": np.asarray(curves["test_naive"]["accuracies"], dtype=np.float64),
        "test_naive_errors": np.asarray(curves["test_naive"]["errors"], dtype=np.float64),
        "train_perm_losses": np.asarray(curves["train_perm"]["losses"], dtype=np.float64),
        "train_perm_accuracies": np.asarray(curves["train_perm"]["accuracies"], dtype=np.float64),
        "train_perm_errors": np.asarray(curves["train_perm"]["errors"], dtype=np.float64),
        "test_perm_losses": np.asarray(curves["test_perm"]["losses"], dtype=np.float64),
        "test_perm_accuracies": np.asarray(curves["test_perm"]["accuracies"], dtype=np.float64),
        "test_perm_errors": np.asarray(curves["test_perm"]["errors"], dtype=np.float64),
    }
    scale_bundle = {
        "lambdas": np.asarray(curves["test_naive"]["lambdas"], dtype=np.float64),
        "train_naive_losses": np.asarray(curves["train_naive"]["losses"], dtype=np.float64),
        "train_naive_accuracies": np.asarray(curves["train_naive"]["accuracies"], dtype=np.float64),
        "train_naive_errors": np.asarray(curves["train_naive"]["errors"], dtype=np.float64),
        "test_naive_losses": np.asarray(curves["test_naive"]["losses"], dtype=np.float64),
        "test_naive_accuracies": np.asarray(curves["test_naive"]["accuracies"], dtype=np.float64),
        "test_naive_errors": np.asarray(curves["test_naive"]["errors"], dtype=np.float64),
        "train_scale_losses": np.asarray(curves["train_scale"]["losses"], dtype=np.float64),
        "train_scale_accuracies": np.asarray(curves["train_scale"]["accuracies"], dtype=np.float64),
        "train_scale_errors": np.asarray(curves["train_scale"]["errors"], dtype=np.float64),
        "test_scale_losses": np.asarray(curves["test_scale"]["losses"], dtype=np.float64),
        "test_scale_accuracies": np.asarray(curves["test_scale"]["accuracies"], dtype=np.float64),
        "test_scale_errors": np.asarray(curves["test_scale"]["errors"], dtype=np.float64),
    }
    np.save(arrays_dir / "perm_bundle.npy", perm_bundle, allow_pickle=True)
    np.save(arrays_dir / "scale_bundle.npy", scale_bundle, allow_pickle=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive, perm-only, and perm+scale interpolation curves for CIFAR10 VGG models.")
    parser.add_argument("--vgg-name", type=str, required=True)
    parser.add_argument("--model-a-checkpoint", type=Path, required=True)
    parser.add_argument("--model-b-checkpoint", type=Path, required=True)
    parser.add_argument("--rebased-perm-checkpoint", type=Path, required=True)
    parser.add_argument("--rebased-scale-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-eval-points", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-plots", action="store_true")
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
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    model_b = load_model_from_checkpoint(
        checkpoint_path=args.model_b_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    rebased_perm = load_model_from_checkpoint(
        checkpoint_path=args.rebased_perm_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    rebased_scale = load_model_from_checkpoint(
        checkpoint_path=args.rebased_scale_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )

    print("Computing naive train curve")
    train_naive = evaluate_curve(
        model_a=model_a,
        model_b=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing naive test curve")
    test_naive = evaluate_curve(
        model_a=model_a,
        model_b=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing perm-only train curve")
    train_perm = evaluate_curve(
        model_a=rebased_perm,
        model_b=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing perm-only test curve")
    test_perm = evaluate_curve(
        model_a=rebased_perm,
        model_b=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing perm+scale train curve")
    train_scale = evaluate_curve(
        model_a=rebased_scale,
        model_b=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing perm+scale test curve")
    test_scale = evaluate_curve(
        model_a=rebased_scale,
        model_b=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )

    curves = {
        "train_naive": train_naive,
        "test_naive": test_naive,
        "train_perm": train_perm,
        "test_perm": test_perm,
        "train_scale": train_scale,
        "test_scale": test_scale,
    }
    save_curve_arrays(output_dir, curves)

    if not args.skip_plots:
        show_legend = str(args.vgg_name).upper() == "VGG11"
        plot_three_curves(
            x=test_naive["lambdas"],
            y_naive=test_naive["losses"],
            y_perm=test_perm["losses"],
            y_scale=test_scale["losses"],
            title=f"{args.vgg_name}: test loss",
            ylabel="Test Loss",
            output_path=output_dir / "compare_test_loss.png",
            show_legend=show_legend,
        )
        plot_three_curves(
            x=test_naive["lambdas"],
            y_naive=test_naive["accuracies"],
            y_perm=test_perm["accuracies"],
            y_scale=test_scale["accuracies"],
            title=f"{args.vgg_name}: test accuracy",
            ylabel="Accuracy (%)",
            output_path=output_dir / "compare_test_accuracy.png",
            show_legend=show_legend,
        )
        plot_three_curves(
            x=test_naive["lambdas"],
            y_naive=test_naive["errors"],
            y_perm=test_perm["errors"],
            y_scale=test_scale["errors"],
            title=f"{args.vgg_name}: test error",
            ylabel="Error (%)",
            output_path=output_dir / "compare_test_error.png",
            show_legend=show_legend,
        )

    payload = {
        "vgg_name": args.vgg_name,
        "model_a_checkpoint": str(args.model_a_checkpoint.resolve()),
        "model_b_checkpoint": str(args.model_b_checkpoint.resolve()),
        "rebased_perm_checkpoint": str(args.rebased_perm_checkpoint.resolve()),
        "rebased_scale_checkpoint": str(args.rebased_scale_checkpoint.resolve()),
        "data_path": str(args.data_path.resolve()),
        "eval_transform": eval_transform,
        "num_eval_points": int(args.num_eval_points),
        "device": str(device),
        "skip_plots": bool(args.skip_plots),
        "arrays_dir": str((output_dir / "arrays").resolve()),
        "curves": curves,
    }
    with open(output_dir / "curves.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
