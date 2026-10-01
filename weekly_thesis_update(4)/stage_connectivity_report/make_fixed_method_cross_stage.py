#!/usr/bin/env python3
"""Plot fixed-method cross-stage test-loss barriers for the calibration pair."""

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
    {
        "slug": "fashion_mnist_mlp",
        "title": "Fashion-MNIST — MLP 10×512",
        "csv": REPO / "results/dense_linear_stage_fashion_mnist_10k_atol2e4/report/barriers.csv",
    },
    {
        "slug": "cifar10_vgg11",
        "title": "CIFAR-10 — VGG11",
        "csv": REPO / "results/dense_linear_stage_vgg11_10k/report/barriers.csv",
    },
)

METHODS = (
    ("sinkhorn", "Sinkhorn permutation only", "sinkhorn"),
    ("sinkhorn_scale_finetune", "Sinkhorn + scale fine-tuning", "sinkhorn_scale"),
    ("wm", "WM + scale", "wm"),
)


def load(path: Path, method: str):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    relevant = [
        row
        for row in rows
        if row["method"] == method
        and row["subset"] == "test_full"
        and row["metric"] == "loss"
        and row["barrier"] == "chord"
    ]
    stages = sorted({int(row["left_epoch"]) for row in relevant})
    final = max(stages)

    def summarize(left: int, right: int):
        dense = [
            row for row in relevant
            if int(row["left_epoch"]) == left
            and int(row["right_epoch"]) == right
            and row["resolution"] == "selected_dense"
        ]
        dense_by_replicate = {int(row["replicate"]): float(row["value"]) for row in dense}
        if len(dense_by_replicate) > 1:
            values = np.asarray(list(dense_by_replicate.values()), dtype=float)
        else:
            coarse = [
                row for row in relevant
                if int(row["replicate"]) == 0
                and int(row["left_epoch"]) == left
                and int(row["right_epoch"]) == right
                and row["resolution"] == "coarse"
            ]
            if len(coarse) != 1:
                raise ValueError(f"Missing unique coarse value for {(left, right, method)}")
            values = np.asarray([float(coarse[0]["value"])], dtype=float)
        return (
            float(values.mean()),
            float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            len(values),
        )

    final_left = [summarize(final, epoch) for epoch in stages]
    final_right = [summarize(epoch, final) for epoch in stages]
    return stages, final, final_left, final_right


def render(experiment, method: str, method_title: str, method_slug: str) -> None:
    stages, final, final_left, final_right = load(experiment["csv"], method)
    fig, axis = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
    for values, label, linestyle, color in (
        (final_left, rf"$A_{{{final}}}$--$B_n$", "-", "#2ca02c"),
        (final_right, rf"$A_n$--$B_{{{final}}}$", "--", "#9467bd"),
    ):
        means = np.asarray([value[0] for value in values])
        stds = np.asarray([value[1] for value in values])
        counts = np.asarray([value[2] for value in values])
        stage_values = np.asarray(stages)
        axis.plot(
            stages, means, linewidth=2.2, linestyle=linestyle,
            color=color, label=label,
        )
        multiple = counts > 1
        single = ~multiple
        if multiple.any():
            axis.errorbar(
                stage_values[multiple], means[multiple], yerr=stds[multiple],
                fmt="o", markersize=6, capsize=3, color=color,
                markerfacecolor=color, markeredgecolor=color, zorder=3,
            )
        if single.any():
            axis.scatter(
                stage_values[single], means[single], s=38, facecolors="white",
                edgecolors=color, linewidths=1.8, zorder=3,
            )
    axis.set_title(f"{experiment['title']}\n{method_title}")
    axis.set_xscale("symlog", linthresh=2.0, linscale=1.0)
    axis.set_xticks(stages, [str(stage) for stage in stages], rotation=45)
    axis.set_xlabel("completed epoch")
    axis.set_ylabel(r"test-loss $B_{\mathrm{chord}}$")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.text(
        0.99, 0.01, r"error bars: $\pm1$ SD when $N>1$; hollow: $N=1$",
        ha="right", va="bottom", fontsize=7, color="0.4",
    )

    output = FIGURES / f"{experiment['slug']}_{method_slug}_cross_stage_test_loss"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_original_figure3_right() -> None:
    path = REPO / "results/dense_linear_stage_vgg11_10k/report/barriers.csv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows = [
        row for row in rows
        if row["selected_overall"] == "True"
        and row["resolution"] == "selected_dense"
        and row["subset"] == "test_full"
        and row["metric"] == "loss"
        and row["barrier"] == "chord"
    ]
    stages = sorted({int(row["left_epoch"]) for row in rows})
    replicates = sorted({int(row["replicate"]) for row in rows})
    final = max(stages)
    lookup = {
        (int(row["replicate"]), int(row["left_epoch"]), int(row["right_epoch"])):
        float(row["value"])
        for row in rows
    }

    fig, axis = plt.subplots(figsize=(6.4, 4.1), constrained_layout=True)
    for pairs, label, linestyle, color in (
        ([(final, epoch) for epoch in stages], rf"$A_{{{final}}}$--$B_n$", "-", "#2ca02c"),
        ([(epoch, final) for epoch in stages], rf"$A_n$--$B_{{{final}}}$", "--", "#9467bd"),
    ):
        matrix = np.asarray([
            [lookup[(replicate, left, right)] for left, right in pairs]
            for replicate in replicates
        ])
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0, ddof=1)
        axis.plot(stages, mean, marker="o", linewidth=2.2, linestyle=linestyle,
                  color=color, label=label)
        axis.fill_between(stages, np.maximum(0.0, mean - std), mean + std,
                          color=color, alpha=0.16)

    axis.set_title("CIFAR-10 — VGG11\nBest overall alignment")
    axis.set_xscale("symlog", linthresh=2.0, linscale=1.0)
    axis.set_xticks(stages, [str(stage) for stage in stages], rotation=45)
    axis.set_xlabel("completed epoch")
    axis.set_ylabel(r"test-loss $B_{\mathrm{chord}}$")
    axis.grid(alpha=0.2)
    axis.legend()
    output = FIGURES / "vgg11_original_figure3_right_test_loss"
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
    for experiment in EXPERIMENTS:
        for method, method_title, method_slug in METHODS:
            render(experiment, method, method_title, method_slug)
    render_original_figure3_right()


if __name__ == "__main__":
    main()
