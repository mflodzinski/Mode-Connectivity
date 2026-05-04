#!/usr/bin/env python3
"""Plot open_lth shared-split interpolation barriers vs L2 distance and split iteration.

This reads the outputs produced by:
    results/vgg16/cifar10/endpoints/open_lth_shared_split/iter{split}/evaluation/interpolation.json

For each split it:
1. loads the saved interpolation summary barriers
2. loads the two endpoint checkpoints referenced in the JSON
3. computes the L2 distance between endpoint weights
4. writes a CSV summary
5. saves barrier-vs-L2 and barrier-vs-iters plots
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent

METRIC_CHOICES = [
    "train_loss_barrier",
    "test_loss_barrier",
    "train_acc_barrier",
    "test_acc_barrier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot open_lth shared-split interpolation barriers vs L2 distance and split iterations."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/vgg16/cifar10/endpoints/open_lth_shared_split",
        help="Root containing iter*/evaluation/interpolation.json",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[0, 25, 100, 500, 1000],
        help="Split iterations to include.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=METRIC_CHOICES,
        choices=METRIC_CHOICES,
        help="Barrier metrics to plot.",
    )
    parser.add_argument(
        "--iters-per-epoch",
        type=float,
        default=391.0,
        help="Used only for the CSV convenience column shared_epochs.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="plots/open_lth_shared_split",
        help="Prefix for plot filenames.",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="plots/open_lth_shared_split_barriers.csv",
        help="CSV summary output path.",
    )
    return parser.parse_args()


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def load_state_dict(path: Path) -> OrderedDict[str, torch.Tensor]:
    obj = safe_torch_load(path)
    if isinstance(obj, dict) and "model_state" in obj:
        obj = obj["model_state"]
    elif isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, (dict, OrderedDict)):
        raise TypeError(f"Unsupported checkpoint object at {path}: {type(obj)}")
    return OrderedDict((k, v.detach().cpu()) for k, v in obj.items())


def compute_l2_distance(state_a: OrderedDict[str, torch.Tensor], state_b: OrderedDict[str, torch.Tensor]) -> float:
    total = 0.0
    for key in state_a:
        tensor_a = state_a[key]
        tensor_b = state_b[key]
        if not torch.is_tensor(tensor_a) or not torch.is_tensor(tensor_b):
            continue
        if not torch.is_floating_point(tensor_a):
            continue
        diff = tensor_a.float() - tensor_b.float()
        total += diff.pow(2).sum().item()
    return total ** 0.5


def metric_label(metric: str) -> str:
    return {
        "train_loss_barrier": "Train Loss Barrier",
        "test_loss_barrier": "Test Loss Barrier",
        "train_acc_barrier": "Train Accuracy Barrier (%)",
        "test_acc_barrier": "Test Accuracy Barrier (%)",
    }[metric]


def plot_vs_l2(rows: list[dict], metric: str, output_path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["l2_distance"])
    xs = [row["l2_distance"] for row in rows]
    ys = [row[metric] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs, ys, color="#1f77b4", lw=1.5, alpha=0.7)
    ax.scatter(xs, ys, color="#1f77b4", s=80, zorder=5)

    for row, x, y in zip(rows, xs, ys):
        ax.annotate(str(row["split_iter"]), (x, y), textcoords="offset points", xytext=(4, 5), fontsize=8)

    ax.set_xlabel("L2 Distance", fontsize=12)
    ax.set_ylabel(metric_label(metric), fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_vs_iters(rows: list[dict], metric: str, output_path: Path) -> None:
    rows = sorted(rows, key=lambda row: row["split_iter"])
    xs = [row["split_iter"] for row in rows]
    ys = [row[metric] for row in rows]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs, ys, color="#d62728", lw=1.8, marker="o", markersize=7)
    ax.set_xlabel("Shared Split Iteration", fontsize=12)
    ax.set_ylabel(metric_label(metric), fontsize=12)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def csv_fieldnames(metrics: Iterable[str]) -> list[str]:
    return ["split_iter", "shared_epochs", "l2_distance", *metrics, "endpoint_a", "endpoint_b"]


def main() -> None:
    args = parse_args()
    results_root = PROJECT_ROOT / args.results_root
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    rows = []
    for split in args.splits:
        json_path = results_root / f"iter{split}" / "evaluation" / "interpolation.json"
        if not json_path.exists():
            print(f"Skipping split {split}: missing {json_path}")
            continue

        with json_path.open("r") as handle:
            payload = json.load(handle)

        summary = payload["interpolation"]["summary"]
        endpoint_paths = [PROJECT_ROOT / rel_path for rel_path in payload["endpoint_paths"]]
        state_a = load_state_dict(endpoint_paths[0])
        state_b = load_state_dict(endpoint_paths[1])
        l2_distance = compute_l2_distance(state_a, state_b)

        row = {
            "split_iter": split,
            "shared_epochs": split / args.iters_per_epoch,
            "l2_distance": l2_distance,
            "endpoint_a": str(endpoint_paths[0].relative_to(PROJECT_ROOT)),
            "endpoint_b": str(endpoint_paths[1].relative_to(PROJECT_ROOT)),
        }
        for metric in METRIC_CHOICES:
            row[metric] = float(summary[metric])
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid open_lth split results found.")

    rows = sorted(rows, key=lambda row: row["split_iter"])

    csv_path = PROJECT_ROOT / args.csv_output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fieldnames(METRIC_CHOICES))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {csv_path}")

    output_prefix = PROJECT_ROOT / args.output_prefix
    for metric in args.metrics:
        plot_vs_l2(rows, metric, output_prefix.parent / f"{output_prefix.name}_{metric}_vs_l2.png")
        plot_vs_iters(rows, metric, output_prefix.parent / f"{output_prefix.name}_{metric}_vs_iters.png")

    print("\n" + "=" * 110)
    print("OPEN_LTH SHARED-SPLIT SUMMARY")
    print("=" * 110)
    header = (
        f"{'split_iter':>10} {'shared_ep':>10} {'l2_distance':>14} "
        f"{'train_loss_bar':>16} {'test_loss_bar':>15} "
        f"{'train_acc_bar':>15} {'test_acc_bar':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['split_iter']:>10d} "
            f"{row['shared_epochs']:>10.4f} "
            f"{row['l2_distance']:>14.6f} "
            f"{row['train_loss_barrier']:>16.6f} "
            f"{row['test_loss_barrier']:>15.6f} "
            f"{row['train_acc_barrier']:>15.6f} "
            f"{row['test_acc_barrier']:>14.6f}"
        )


if __name__ == "__main__":
    import pickle

    main()
