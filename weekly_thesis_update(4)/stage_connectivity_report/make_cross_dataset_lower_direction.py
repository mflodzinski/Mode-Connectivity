#!/usr/bin/env python3
"""Plot both barriers for the lower-barrier A_final--B_n direction."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIGURES = HERE / "figures"

EXPERIMENTS = (
    (
        "fashion_mnist_mlp",
        "Fashion-MNIST — MLP 10×512",
        REPO / "results/dense_linear_stage_fashion_mnist_10k_atol2e4/report/barriers.csv",
    ),
    (
        "cifar10_vgg11",
        "CIFAR-10 — VGG11",
        REPO / "results/dense_linear_stage_vgg11_10k/report/barriers.csv",
    ),
)


def load(path: Path, barrier: str):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows = [
        row for row in rows
        if row["selected_overall"] == "True"
        and row["resolution"] == "selected_dense"
        and row["subset"] == "test_full"
        and row["metric"] == "loss"
        and row["barrier"] == barrier
    ]
    stages = sorted({int(row["left_epoch"]) for row in rows})
    replicates = sorted({int(row["replicate"]) for row in rows})
    final = max(stages)
    lookup = {
        (int(row["replicate"]), int(row["left_epoch"]), int(row["right_epoch"])):
        float(row["value"])
        for row in rows
    }
    matrix = np.asarray([
        [lookup[(replicate, final, epoch)] for epoch in stages]
        for replicate in replicates
    ])
    return stages, matrix.mean(axis=0), matrix.std(axis=0, ddof=1), final


def render(slug: str, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    for barrier, label, color, linestyle in (
        ("chord", r"$B_{\mathrm{chord}}$", "#1f77b4", "-"),
        ("worse", r"$B_{\mathrm{worse}}$", "#d62728", "--"),
    ):
        stages, mean, std, final = load(path, barrier)
        axis.plot(
            stages, mean, marker="o", linewidth=2.2, linestyle=linestyle,
            color=color, label=label,
        )
        axis.fill_between(
            stages, np.maximum(0.0, mean - std), mean + std,
            color=color, alpha=0.16,
        )

    axis.set_title(f"{title}\n" + rf"Best overall, $A_{{{final}}}$--$B_n$")
    axis.set_xscale("symlog", linthresh=2.0, linscale=1.0)
    axis.set_xticks(stages, [str(stage) for stage in stages], rotation=45)
    axis.set_xlabel("completed epoch of $B_n$")
    axis.set_ylabel("test-loss barrier")
    axis.grid(alpha=0.2)
    axis.legend()

    output = FIGURES / f"{slug}_overall_afinal_bn_test_loss_barriers"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_combined() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.25), constrained_layout=True)
    for axis, (_, title, path) in zip(axes, EXPERIMENTS):
        for barrier, label, color, linestyle in (
            ("chord", r"$B_{\mathrm{chord}}$", "#1f77b4", "-"),
            ("worse", r"$B_{\mathrm{worse}}$", "#d62728", "--"),
        ):
            stages, mean, std, final = load(path, barrier)
            axis.plot(
                stages, mean, marker="o", linewidth=2.2, linestyle=linestyle,
                color=color, label=label,
            )
            axis.fill_between(
                stages, np.maximum(0.0, mean - std), mean + std,
                color=color, alpha=0.16,
            )
        axis.set_title(f"{title}\n" + rf"Best overall, $A_{{{final}}}$--$B_n$")
        axis.set_xscale("symlog", linthresh=2.0, linscale=1.0)
        axis.set_xticks(stages, [str(stage) for stage in stages], rotation=45)
        axis.set_xlabel("completed epoch of $B_n$")
        axis.set_ylabel("test-loss barrier")
        axis.grid(alpha=0.2)
        axis.legend()

    output = FIGURES / "overall_afinal_bn_test_loss_barriers_by_architecture"
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "figure.dpi": 140,
    })
    for slug, title, path in EXPERIMENTS:
        render(slug, title, path)
    render_combined()


if __name__ == "__main__":
    main()
