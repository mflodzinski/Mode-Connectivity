#!/usr/bin/env python3
"""Plot three loss-barrier regimes from evaluated interpolation files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(
    curve_npz: Path,
    linear_npz: Path,
    metric: str,
    middle_divisor: float,
    output_path: Path,
    ylabel_fontsize: float,
    xlabel_fontsize: float,
    legend_fontsize: float,
) -> None:
    curve_data = np.load(curve_npz)
    linear_data = np.load(linear_npz)

    if metric not in curve_data.files:
        raise KeyError(f"Metric '{metric}' not found in {curve_npz}")
    if metric not in linear_data.files:
        raise KeyError(f"Metric '{metric}' not found in {linear_npz}")

    ts_curve = curve_data["ts"]
    ts_linear = linear_data["ts"]
    if ts_curve.shape != ts_linear.shape or not np.allclose(ts_curve, ts_linear):
        raise ValueError("curve.npz and linear.npz have incompatible 'ts' arrays")

    lambdas = ts_curve
    low_barrier_curve = np.asarray(curve_data[metric], dtype=float)
    high_barrier_curve = np.asarray(linear_data[metric], dtype=float)
    mid_barrier_curve = high_barrier_curve / middle_divisor

    fig, ax = plt.subplots(figsize=(9.8, 5.8))

    ax.plot(
        lambdas,
        low_barrier_curve,
        color="#1b9e77",
        lw=2.4,
        linestyle="-",
        label=r"Barrier $\approx 0$",
    )
    ax.plot(
        lambdas,
        mid_barrier_curve,
        color="#e6ab02",
        lw=2.4,
        linestyle="-",
        label=r"Barrier $> 0$",
    )
    ax.plot(
        lambdas,
        high_barrier_curve,
        color="#d95f02",
        lw=2.6,
        linestyle="-",
        label=r"Barrier $\approx$ initialization-level loss",
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("t (interpolation parameter)", fontsize=xlabel_fontsize)
    ax.set_ylabel("Loss", fontsize=ylabel_fontsize, fontweight="bold")
    ax.grid(alpha=0.25)
    ax.legend(loc="center", frameon=True, fontsize=legend_fontsize)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=240)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve-npz",
        type=Path,
        default=Path("results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/curve.npz"),
        help="Path to evaluated curve file.",
    )
    parser.add_argument(
        "--linear-npz",
        type=Path,
        default=Path("results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/linear.npz"),
        help="Path to evaluated linear interpolation file.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tr_loss",
        help="Metric key present in both npz files, for example te_loss or tr_loss.",
    )
    parser.add_argument(
        "--middle-divisor",
        type=float,
        default=5.0,
        help="Divisor for the middle-threshold curve derived from linear.npz.",
    )
    parser.add_argument(
        "--ylabel-fontsize",
        type=float,
        default=16.0,
        help="Font size for the y-axis title.",
    )
    parser.add_argument(
        "--xlabel-fontsize",
        type=float,
        default=13.0,
        help="Font size for the x-axis title.",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=float,
        default=15.0,
        help="Font size for the legend text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/loss_barrier_thresholds_from_eval.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw(
        args.curve_npz,
        args.linear_npz,
        args.metric,
        args.middle_divisor,
        args.output,
        args.ylabel_fontsize,
        args.xlabel_fontsize,
        args.legend_fontsize,
    )
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
