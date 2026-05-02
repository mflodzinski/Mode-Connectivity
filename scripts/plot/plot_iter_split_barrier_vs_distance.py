"""
Plot LMC barrier vs L2 distance for iteration-split shared-training pairs.

This script reads the raw pair checkpoints produced by:
    results/vgg16/cifar10/endpoints/lmc_connected_iter_noaug/iter{split}/seed{0,1}/checkpoint-200.pt

and computes, for each split:
- L2 distance between the two endpoint models
- linear interpolation barrier

By default the plotted barrier is the relative test-loss barrier:
    (max_t test_loss(t) - endpoint_avg) / (endpoint_avg + eps)

It also writes a CSV summary with the key metrics for each split.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

EPS = 1e-12

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.benchmark_alignment import (  # noqa: E402
    compute_l2_distance,
    evaluate_barrier,
    load_model,
)

sys.path.insert(0, str(PROJECT_ROOT / "external" / "dnn-mode-connectivity"))
import data  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LMC barrier vs L2 distance for iteration-split shared-training pairs."
    )
    parser.add_argument(
        "--endpoints-root",
        type=str,
        default="results/vgg16/cifar10/endpoints/lmc_connected_iter_noaug",
        help="Root containing iter{split}/seed{0,1}/checkpoint-200.pt",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[0, 25, 100, 1000, 5000],
        help="Split iterations to include",
    )
    parser.add_argument("--data-path", type=str, default="./data", help="Dataset path")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument(
        "--num-eval-points",
        type=int,
        default=61,
        help="Number of interpolation points for barrier evaluation",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="test_loss_barrier_rel",
        choices=[
            "test_loss_barrier",
            "train_loss_barrier",
            "test_loss_barrier_rel",
            "train_loss_barrier_rel",
            "test_acc_barrier_rel",
            "train_acc_barrier_rel",
            "min_test_acc",
            "min_train_acc",
        ],
        help="Which quantity to plot on the y-axis",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        choices=[
            "test_loss_barrier",
            "train_loss_barrier",
            "test_loss_barrier_rel",
            "train_loss_barrier_rel",
            "test_acc_barrier_rel",
            "train_acc_barrier_rel",
            "min_test_acc",
            "min_train_acc",
        ],
        help="Optional list of metrics to plot in one pass. If set, overrides --metric.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/lmc_iter_noaug_barrier_vs_distance.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="plots/lmc_iter_noaug",
        help="Prefix for multi-metric plot outputs when using --metrics",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="plots/lmc_iter_noaug_barrier_vs_distance.csv",
        help="Output CSV summary path",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    return parser.parse_args()


def load_eval_loaders(data_path: str, batch_size: int, num_workers: int):
    loaders, _ = data.loaders(
        "CIFAR10",
        data_path,
        batch_size,
        num_workers=num_workers,
        transform_name="VGG",
        use_test=True,
        eval_mode=True,
    )
    return loaders


def y_value(row: dict, metric: str) -> float:
    if metric == "test_loss_barrier":
        return row["test_loss_barrier"]
    if metric == "train_loss_barrier":
        return row["train_loss_barrier"]
    if metric == "test_loss_barrier_rel":
        return row["test_loss_barrier_rel"]
    if metric == "train_loss_barrier_rel":
        return row["train_loss_barrier_rel"]
    if metric == "test_acc_barrier_rel":
        return row["test_acc_barrier_rel"]
    if metric == "train_acc_barrier_rel":
        return row["train_acc_barrier_rel"]
    if metric == "min_test_acc":
        return row["min_test_acc"]
    if metric == "min_train_acc":
        return row["min_train_acc"]
    raise ValueError(f"Unsupported metric: {metric}")


def y_label(metric: str) -> str:
    return {
        "test_loss_barrier": "Test Loss Barrier",
        "train_loss_barrier": "Train Loss Barrier",
        "test_loss_barrier_rel": "Relative Test Loss Barrier",
        "train_loss_barrier_rel": "Relative Train Loss Barrier",
        "test_acc_barrier_rel": "Relative Test Accuracy Barrier",
        "train_acc_barrier_rel": "Relative Train Accuracy Barrier",
        "min_test_acc": "Minimum Test Accuracy (%)",
        "min_train_acc": "Minimum Train Accuracy (%)",
    }[metric]


def save_plot(rows: list[dict], metric: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    xs = [row["l2_distance"] for row in rows]
    ys = [y_value(row, metric) for row in rows]

    ax.plot(xs, ys, color="#1f77b4", alpha=0.65, lw=1.5)
    ax.scatter(xs, ys, color="#1f77b4", s=80, zorder=5)

    for row, x, y in zip(rows, xs, ys):
        ax.annotate(
            str(row["split_iter"]),
            (x, y),
            textcoords="offset points",
            xytext=(4, 5),
            fontsize=8,
        )

    ax.set_xlabel("L2 Distance", fontsize=12)
    ax.set_ylabel(y_label(metric), fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def endpoint_avg(values):
    return 0.5 * (values[0] + values[-1])


def loss_barrier_abs(values):
    return max(values) - endpoint_avg(values)


def loss_barrier_rel(values, eps=EPS):
    avg = endpoint_avg(values)
    return (max(values) - avg) / (avg + eps)


def acc_barrier_rel(values, eps=EPS):
    avg = endpoint_avg(values)
    return (avg - min(values)) / (avg + eps)


def main() -> None:
    args = parse_args()

    endpoints_root = PROJECT_ROOT / args.endpoints_root
    if not endpoints_root.exists():
        raise FileNotFoundError(f"Endpoints root not found: {endpoints_root}")

    loaders = load_eval_loaders(args.data_path, args.batch_size, args.num_workers)

    rows = []
    for split in args.splits:
        ckpt0 = endpoints_root / f"iter{split}" / "seed0" / "checkpoint-200.pt"
        ckpt1 = endpoints_root / f"iter{split}" / "seed1" / "checkpoint-200.pt"
        if not ckpt0.exists() or not ckpt1.exists():
            print(f"Skipping split {split}: missing checkpoint(s)")
            continue

        model0, state0, _ = load_model(str(ckpt0))
        model1, state1, _ = load_model(str(ckpt1))

        dist = compute_l2_distance(state0, state1)
        barrier = evaluate_barrier(model0, model1, loaders, args.num_eval_points)

        train_loss_barrier = loss_barrier_abs(barrier["train_loss"])
        test_loss_barrier = loss_barrier_abs(barrier["test_loss"])
        train_loss_barrier_rel = loss_barrier_rel(barrier["train_loss"])
        test_loss_barrier_rel = loss_barrier_rel(barrier["test_loss"])
        train_acc_barrier_rel = acc_barrier_rel(barrier["train_acc"])
        test_acc_barrier_rel = acc_barrier_rel(barrier["test_acc"])

        row = {
            "split_iter": split,
            "l2_distance": dist["l2_distance"],
            "rms_difference": dist["rms_difference"],
            "train_loss_barrier": train_loss_barrier,
            "test_loss_barrier": test_loss_barrier,
            "train_loss_barrier_rel": train_loss_barrier_rel,
            "test_loss_barrier_rel": test_loss_barrier_rel,
            "train_acc_barrier_rel": train_acc_barrier_rel,
            "test_acc_barrier_rel": test_acc_barrier_rel,
            "min_train_acc": min(barrier["train_acc"]),
            "min_test_acc": min(barrier["test_acc"]),
            "endpoint_avg_train_loss": endpoint_avg(barrier["train_loss"]),
            "endpoint_avg_test_loss": barrier["endpoint_avg_test_loss"],
            "endpoint_avg_train_acc": endpoint_avg(barrier["train_acc"]),
            "endpoint_avg_test_acc": endpoint_avg(barrier["test_acc"]),
            "max_train_loss": max(barrier["train_loss"]),
            "max_test_loss": barrier["max_test_loss"],
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No valid split pairs found.")

    rows.sort(key=lambda r: r["split_iter"])

    csv_path = PROJECT_ROOT / args.csv_output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split_iter",
                "l2_distance",
                "rms_difference",
                "train_loss_barrier",
                "test_loss_barrier",
                "train_loss_barrier_rel",
                "test_loss_barrier_rel",
                "train_acc_barrier_rel",
                "test_acc_barrier_rel",
                "min_train_acc",
                "min_test_acc",
                "endpoint_avg_train_loss",
                "endpoint_avg_test_loss",
                "endpoint_avg_train_acc",
                "endpoint_avg_test_acc",
                "max_train_loss",
                "max_test_loss",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV to {csv_path}")

    metrics_to_plot = args.metrics if args.metrics else [args.metric]
    if args.metrics:
        for metric in metrics_to_plot:
            output_path = PROJECT_ROOT / f"{args.output_prefix}_{metric}_vs_distance.png"
            save_plot(rows, metric, output_path)
    else:
        output_path = PROJECT_ROOT / args.output
        save_plot(rows, args.metric, output_path)

    print("\n" + "=" * 95)
    print("ITER-SPLIT LMC SUMMARY")
    print("=" * 95)
    print(
        f"{'split_iter':>10} {'l2_distance':>14} {'test_loss_bar':>14} "
        f"{'train_loss_bar':>15} {'test_loss_rel':>14} {'train_loss_rel':>15} "
        f"{'test_acc_rel':>13} {'train_acc_rel':>14}"
    )
    print("-" * 95)
    for row in rows:
        print(
            f"{row['split_iter']:>10d} "
            f"{row['l2_distance']:>14.4f} "
            f"{row['test_loss_barrier']:>14.6f} "
            f"{row['train_loss_barrier']:>15.6f} "
            f"{row['test_loss_barrier_rel']:>14.6f} "
            f"{row['train_loss_barrier_rel']:>15.6f} "
            f"{row['test_acc_barrier_rel']:>13.6f} "
            f"{row['train_acc_barrier_rel']:>14.6f}"
        )

    if args.show and not args.metrics:
        plt.show()


if __name__ == "__main__":
    main()
