"""Evaluate linear interpolation for one retained pytorch-vgg checkpoint pair.

The script loads two endpoint checkpoints, evaluates the interpolation grid,
and writes the metrics consumed by the shared-split benchmark plots.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mode_connectivity.core import data as core_data
from mode_connectivity.core.checkpoint import build_model_from_state_dict, load_checkpoint_state
from mode_connectivity.common.paths import PROJECT_ROOT as REPO_ROOT
from mode_connectivity.evaluation.interpolation import evaluate_linear_interpolation, summarize_interpolation_metrics
from mode_connectivity.evaluation.metrics import state_distance_summary

PROJECT_ROOT = REPO_ROOT


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

    dist = state_distance_summary(state_w0, state_w1)
    model = build_model_from_state_dict(
        state_w0,
        checkpoint_family="pytorch_vgg_cifar10",
        num_classes=10,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    train_loader, test_loader = core_data.build_cifar10_pytorch_vgg_eval_loaders(
        args.data_root,
        args.batch_size,
        args.workers,
    )

    ts = np.linspace(0.0, 1.0, args.num_points)
    interpolation = evaluate_linear_interpolation(
        state_a=state_w0,
        state_b=state_w1,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        ts=ts,
    )

    payload = {
        "w0": str(w0_path.relative_to(PROJECT_ROOT)),
        "w1": str(w1_path.relative_to(PROJECT_ROOT)),
        "distance": dist,
        "interpolation": interpolation,
        "summary": summarize_interpolation_metrics(interpolation),
    }

    with (output_dir / "interpolation.json").open("w") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(output_dir / "interpolation.csv", interpolation)

    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(dist, indent=2))
    print(f"Saved evaluation to {output_dir}")


if __name__ == "__main__":
    main()
