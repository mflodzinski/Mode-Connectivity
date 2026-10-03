"""Regenerate the three XOR interpolation panels used in the AISTATS paper."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mode_connectivity.xor.xor_permutation_scale_experiment import (
    get_core_plot_methods,
    get_curve_linestyles,
    get_plot_style,
)


PAPER_FIGURES = ROOT / "weekly_thesis_update(4)" / "paper_aistats2027" / "figures"
OUTPUT_NAMES = {3: "3h_xor.png", 5: "h5_xor.png", 7: "h7_xor.png"}


def load_aggregates(width: int) -> dict[str, dict[str, list[float]]]:
    results_path = (
        ROOT
        / "results"
        / "xor"
        / f"xor_{width}h_perm_vs_scale"
        / "xor_perm_scale_results.json"
    )
    with results_path.open(encoding="utf-8") as handle:
        return json.load(handle)["aggregate_curves"]


def plot_panel(width: int) -> None:
    aggregates = load_aggregates(width)
    colors, labels = get_plot_style()
    linestyles = get_curve_linestyles()
    figure, axis = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)

    for method_key in get_core_plot_methods():
        curve = aggregates[method_key]
        axis.plot(
            curve["t"],
            curve["loss_mean"],
            color=colors[method_key],
            label=labels[method_key],
            linestyle=linestyles[method_key],
            linewidth=2.7 if method_key == "perm_plus_scale" else 2.3,
        )

    axis.set_xlabel(r"$\lambda$", fontsize=18)
    # The three panels are displayed side by side, so a single y-axis label on
    # the leftmost panel is sufficient and leaves more room for the curves.
    if width == 3:
        axis.set_ylabel("Loss", fontsize=18)
    axis.set_title(f"Hidden size {width}", fontsize=20, fontweight="bold")
    axis.set_xticks(np.linspace(0.0, 1.0, 5))
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    axis.tick_params(axis="both", labelsize=15, width=1.2, length=5)
    for spine in axis.spines.values():
        spine.set_linewidth(1.2)

    if width == 5:
        axis.legend(
            loc="center",
            bbox_to_anchor=(0.5, 0.62),
            fontsize=15,
            frameon=True,
            framealpha=0.9,
            edgecolor="0.75",
        )

    figure.savefig(PAPER_FIGURES / OUTPUT_NAMES[width], dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    for width in OUTPUT_NAMES:
        plot_panel(width)


if __name__ == "__main__":
    main()
