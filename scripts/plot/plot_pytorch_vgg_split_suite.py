"""Plot barrier vs distance and barrier vs shared epochs for pytorch-vgg shared-split runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PAIR_SPECS = [
    ("100/100", 100.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_100split"),
    ("80/120", 80.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_80split"),
    ("30/170", 30.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_30split"),
    ("8/192", 8.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_8split"),
    ("6/194", 6.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_6split"),
    ("3/197", 3.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_3split"),
    ("2/198", 2.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_2split"),
    ("1/199", 1.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_1split"),
    ("0/200", 0.0, "results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_0split"),
    ("independent", None, "results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing"),
]

METRICS = [
    "train_loss_barrier",
    "test_loss_barrier",
    "train_acc_barrier",
    "test_acc_barrier",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pytorch-vgg shared-split barriers.")
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="plots/pytorch_vgg_split_suite",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="plots/pytorch_vgg_split_suite.csv",
    )
    return parser.parse_args()


def metric_label(metric: str) -> str:
    return {
        "train_loss_barrier": "Train Loss Barrier",
        "test_loss_barrier": "Test Loss Barrier",
        "train_acc_barrier": "Train Accuracy Barrier (%)",
        "test_acc_barrier": "Test Accuracy Barrier (%)",
    }[metric]


def load_rows() -> list[dict]:
    rows = []
    for label, shared_epochs, root_rel in PAIR_SPECS:
        json_path = PROJECT_ROOT / root_rel / "evaluation" / "interpolation.json"
        if not json_path.exists():
            print(f"Skipping {label}: missing {json_path}")
            continue
        with json_path.open("r") as handle:
            payload = json.load(handle)
        distance = payload["distance"]["l2_distance"] if isinstance(payload["distance"], dict) else payload["distance"]
        row = {
            "label": label,
            "shared_epochs": shared_epochs,
            "l2_distance": float(distance),
        }
        for metric in METRICS:
            row[metric] = float(payload["summary"][metric])
        rows.append(row)
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "shared_epochs", "l2_distance", *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {path}")


def plot_vs_distance(rows: list[dict], metric: str, output_path: Path) -> None:
    shared = [row for row in rows if row["shared_epochs"] is not None]
    independent = [row for row in rows if row["shared_epochs"] is None]

    shared = sorted(shared, key=lambda row: row["l2_distance"])
    xs = [row["l2_distance"] for row in shared]
    ys = [row[metric] for row in shared]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs, ys, color="#1f77b4", lw=1.5, alpha=0.7)
    ax.scatter(xs, ys, color="#1f77b4", s=80, zorder=5, label="Shared-split")

    for row in shared:
        ax.annotate(row["label"], (row["l2_distance"], row[metric]), textcoords="offset points", xytext=(4, 5), fontsize=8)

    for row in independent:
        ax.scatter(
            [row["l2_distance"]],
            [row[metric]],
            color="#d62728",
            marker="D",
            s=120,
            edgecolors="black",
            linewidths=1.0,
            zorder=6,
            label="Independent",
        )
        ax.annotate(row["label"], (row["l2_distance"], row[metric]), textcoords="offset points", xytext=(4, 5), fontsize=8)

    ax.set_xlabel("L2 Distance", fontsize=12)
    ax.set_ylabel(metric_label(metric), fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_vs_epochs(rows: list[dict], metric: str, output_path: Path) -> None:
    shared = [row for row in rows if row["shared_epochs"] is not None]
    shared = sorted(shared, key=lambda row: row["shared_epochs"])

    xs = [row["shared_epochs"] for row in shared]
    ys = [row[metric] for row in shared]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(xs, ys, color="#d62728", lw=1.8, marker="o", markersize=7)
    for row in shared:
        ax.annotate(row["label"], (row["shared_epochs"], row[metric]), textcoords="offset points", xytext=(4, 5), fontsize=8)
    ax.set_xlabel("Shared Epochs", fontsize=12)
    ax.set_ylabel(metric_label(metric), fontsize=12)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()
    rows = load_rows()
    if not rows:
        raise RuntimeError("No evaluation files found for pytorch-vgg split suite.")

    write_csv(rows, PROJECT_ROOT / args.csv_output)
    prefix = PROJECT_ROOT / args.output_prefix
    for metric in METRICS:
        plot_vs_distance(rows, metric, prefix.parent / f"{prefix.name}_{metric}_vs_distance.png")
        plot_vs_epochs(rows, metric, prefix.parent / f"{prefix.name}_{metric}_vs_shared_epochs.png")


if __name__ == "__main__":
    main()
