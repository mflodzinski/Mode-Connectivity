"""Evaluate linear interpolation for a pair of pytorch-vgg-cifar10 VGG16 checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from scripts.analysis.benchmark_alignment import build_model_from_state_dict, compute_l2_distance, load_checkpoint_state


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate linear interpolation for a pytorch-vgg-cifar10 checkpoint pair.")
    parser.add_argument("--w0", type=str, required=True, help="Path to first checkpoint")
    parser.add_argument("--w1", type=str, required=True, help="Path to second checkpoint")
    parser.add_argument("--data-root", type=str, default="./data", help="CIFAR10 data root")
    parser.add_argument("--batch-size", type=int, default=128, help="Eval batch size")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--num-points", type=int, default=61, help="Interpolation evaluation points")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write interpolation outputs")
    return parser.parse_args()


def build_eval_loaders(data_root: str, batch_size: int, workers: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=True, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root=data_root, train=False, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            total_correct += outputs.argmax(dim=1).eq(targets).sum().item()
            total += targets.size(0)

    return {"loss": total_loss / total, "accuracy": 100.0 * total_correct / total}


def loss_barrier(values):
    endpoint_avg = 0.5 * (values[0] + values[-1])
    return max(values) - endpoint_avg


def acc_barrier(values):
    endpoint_avg = 0.5 * (values[0] + values[-1])
    return endpoint_avg - min(values)


def write_csv(path: Path, interpolation: dict) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "train_loss", "train_acc", "test_loss", "test_acc"])
        for row in zip(
            interpolation["t"],
            interpolation["train_loss"],
            interpolation["train_acc"],
            interpolation["test_loss"],
            interpolation["test_acc"],
        ):
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    w0_path = PROJECT_ROOT / args.w0
    w1_path = PROJECT_ROOT / args.w1
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    state_w0, fmt0 = load_checkpoint_state(str(w0_path))
    state_w1, fmt1 = load_checkpoint_state(str(w1_path))
    if fmt0 != "pytorch_vgg_cifar10" or fmt1 != "pytorch_vgg_cifar10":
        raise ValueError(f"Expected pytorch_vgg_cifar10 checkpoints, got {fmt0} and {fmt1}")

    dist = compute_l2_distance(state_w0, state_w1)
    model = build_model_from_state_dict(state_w0, "pytorch_vgg_cifar10", num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    train_loader, test_loader = build_eval_loaders(args.data_root, args.batch_size, args.workers)

    ts = np.linspace(0.0, 1.0, args.num_points)
    interpolation = {
        "t": ts.tolist(),
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for idx, t in enumerate(ts):
        print(f"[interp] {idx + 1}/{len(ts)} t={t:.4f}")
        state_t = OrderedDict()
        for key in state_w0:
            state_t[key] = (1.0 - t) * state_w0[key] + t * state_w1[key]
        model.load_state_dict(state_t)

        train_res = evaluate_model(model, train_loader, device)
        test_res = evaluate_model(model, test_loader, device)
        interpolation["train_loss"].append(train_res["loss"])
        interpolation["train_acc"].append(train_res["accuracy"])
        interpolation["test_loss"].append(test_res["loss"])
        interpolation["test_acc"].append(test_res["accuracy"])

    payload = {
        "w0": str(w0_path.relative_to(PROJECT_ROOT)),
        "w1": str(w1_path.relative_to(PROJECT_ROOT)),
        "distance": dist,
        "interpolation": interpolation,
        "summary": {
            "train_loss_barrier": loss_barrier(interpolation["train_loss"]),
            "test_loss_barrier": loss_barrier(interpolation["test_loss"]),
            "train_acc_barrier": acc_barrier(interpolation["train_acc"]),
            "test_acc_barrier": acc_barrier(interpolation["test_acc"]),
            "min_train_acc": min(interpolation["train_acc"]),
            "min_test_acc": min(interpolation["test_acc"]),
        },
    }

    with (output_dir / "interpolation.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(output_dir / "interpolation.csv", interpolation)

    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(dist, indent=2))
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
