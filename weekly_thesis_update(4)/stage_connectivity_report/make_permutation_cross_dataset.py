#!/usr/bin/env python3
"""Compare cross-stage test-loss barriers across datasets."""

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
        "Fashion-MNIST — MLP 10×512",
        REPO / "results/dense_linear_stage_fashion_mnist_10k_atol2e4/report/barriers.csv",
    ),
    (
        "CIFAR-10 — VGG11",
        REPO / "results/dense_linear_stage_vgg11_10k/report/barriers.csv",
    ),
)


def load_series(path: Path, selection: str):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))

    selected = [
        row
        for row in rows
        if row[f"selected_{selection}"] == "True"
        and row["resolution"] == "selected_dense"
        and row["subset"] == "test_full"
        and row["metric"] == "loss"
        and row["barrier"] == "chord"
    ]
    stages = sorted({int(row["left_epoch"]) for row in selected})
    replicates = sorted({int(row["replicate"]) for row in selected})
    final = max(stages)
    lookup = {
        (int(row["replicate"]), int(row["left_epoch"]), int(row["right_epoch"])):
        float(row["value"])
        for row in selected
    }

    def summarize(pairs):
        values = np.asarray(
            [[lookup[(replicate, left, right)] for left, right in pairs]
             for replicate in replicates]
        )
        return values.mean(axis=0), values.std(axis=0, ddof=1)

    final_left = summarize([(final, epoch) for epoch in stages])
    final_right = summarize([(epoch, final) for epoch in stages])
    return stages, final, final_left, final_right


def render(selection: str, heading: str, output_stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    colors = ("#1f77b4", "#ff7f0e")

    for axis, (title, path) in zip(axes, EXPERIMENTS):
        stages, final, final_left, final_right = load_series(path, selection)
        for mean, std, label, style, color in (
            (*final_left, rf"$A_{{{final}}}$--$B_n$", "-", colors[0]),
            (*final_right, rf"$A_n$--$B_{{{final}}}$", "--", colors[1]),
        ):
            axis.plot(
                stages, mean, marker="o", linewidth=2.1, linestyle=style,
                color=color, label=label,
            )
            axis.fill_between(
                stages, np.maximum(0.0, mean - std), mean + std,
                color=color, alpha=0.16,
            )

        axis.set_title(title)
        axis.set_xscale("symlog", linthresh=2.0, linscale=1.0)
        axis.set_xticks(stages, [str(stage) for stage in stages], rotation=45)
        axis.set_xlabel("completed epoch")
        axis.set_ylabel(r"test-loss $B_{\mathrm{chord}}$")
        axis.grid(alpha=0.2)
        axis.legend()

    fig.suptitle(heading)
    output = FIGURES / output_stem
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 140,
    })
    render(
        "permutation_only",
        "Permutation-only cross-stage linear connectivity",
        "permutation_only_cross_stage_test_loss_fashion_vgg11",
    )
    render(
        "overall",
        "Best-overall cross-stage linear connectivity",
        "overall_cross_stage_test_loss_fashion_vgg11",
    )


if __name__ == "__main__":
    main()
