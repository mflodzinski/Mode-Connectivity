"""Aggregate three-way interpolation barriers across retained VGG architectures.

The script reads per-architecture comparison artifacts and reduces them into
the barrier plots used to summarize no-alignment, permutation-only, and
scale-aware alignment performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mode_connectivity.common.paths import PROJECT_ROOT

from mode_connectivity.alignment.permutation_pipeline import compute_paper_loss_barrier


METHOD_SPECS = [
    ("test_naive", "No alignment"),
    ("test_perm", "Permutation only (Sinkhorn)"),
    ("test_scale", "Permutation + scale (joint Sinkhorn)"),
    (
        "perm_then_scale_only",
        "Sinkhorn + scale refinement",
    ),
]

ARCHITECTURES = ["vgg11", "vgg13", "vgg16", "vgg19"]

# Match the Okabe--Ito-derived method colors used by the XOR profiles in
# Figure 2.  Hatching supplies a second, grayscale-safe encoding: both
# scale-aware variants are hatched, while raw and permutation-only bars are
# solid.
METHOD_STYLES = {
    "No alignment": {"color": "#7A7A7A", "hatch": ""},
    "Permutation only (Sinkhorn)": {"color": "#E69F00", "hatch": ""},
    "Permutation + scale (joint Sinkhorn)": {
        "color": "#CC79A7",
        "hatch": "///",
    },
    "Sinkhorn + scale refinement": {
        "color": "#009E73",
        "hatch": "///",
    },
}

PERM_THEN_SCALE_COMPARISON_PATHS = {
    "vgg11": PROJECT_ROOT
    / "results/vgg11/cifar10/raw_pth_align_sweep_perm_then_scale_only/steps50_tau1p0_lr0p05_l1p0_lossmidpoint_lam0p001_ftscale_only_fixed_hard/comparison.json",
    "vgg13": PROJECT_ROOT
    / "results/vgg13/cifar10/raw_pth_align_sweep_perm_then_scale_only_cor_def/steps50_tau1p0_lr0p02_l1p0_lossmidpoint_lam0p001_ftscale_only_fixed_hard/comparison.json",
    "vgg16": PROJECT_ROOT
    / "results/vgg16/cifar10/raw_pth_align_sweep_perm_then_scale_only/steps50_tau1p0_lr0p1_l1p0_lossmidpoint_lam0p001_ftscale_only_fixed_hard/comparison.json",
    "vgg19": PROJECT_ROOT
    / "results/vgg19/cifar10/raw_pth_align_sweep_perm_then_scale_only/steps50_tau1p0_lr0p1_l1p0_lossmidpoint_lam0p001_ftscale_only_fixed_hard/comparison.json",
}


def load_curves(architecture: str) -> dict:
    path = PROJECT_ROOT / "results" / architecture / "cifar10" / "interpolation_comparison_three_way" / "curves.json"
    with open(path, "r") as handle:
        return json.load(handle)


def compute_barrier(losses: list[float], ts: list[float]) -> float:
    return float(compute_paper_loss_barrier(np.asarray(losses, dtype=np.float64), np.asarray(ts, dtype=np.float64)))


def load_perm_then_scale_barrier(architecture: str) -> float:
    path = PERM_THEN_SCALE_COMPARISON_PATHS[architecture]
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison.json for {architecture}: {path}")
    with open(path, "r") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected comparison.json to contain a list at {path}")
    for row in payload:
        if row.get("variant_key") == "original_sinkhorn_lmc":
            return float(row["test_loss_barrier_max_endpoint"])
    raise ValueError(f"comparison.json at {path} does not contain 'original_sinkhorn_lmc'")


def main() -> None:
    plot_data: dict[str, list[float]] = {label: [] for _, label in METHOD_SPECS}
    architecture_labels: list[str] = []

    for architecture in ARCHITECTURES:
        payload = load_curves(architecture)
        architecture_labels.append(str(payload["vgg_name"]))
        curves = payload["curves"]
        for curve_key, label in METHOD_SPECS[:3]:
            curve = curves[curve_key]
            plot_data[label].append(compute_barrier(curve["losses"], curve["lambdas"]))
        plot_data["Sinkhorn + scale refinement"].append(
            load_perm_then_scale_barrier(architecture)
        )

    output_root = PROJECT_ROOT / "results" / "vgg_cifar10_three_way_barriers"
    output_root.mkdir(parents=True, exist_ok=True)
    thesis_output_root = PROJECT_ROOT / "thesis" / "figures" / "new"
    thesis_output_root.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(architecture_labels))
    width = 0.14

    def save_barplot(output_path: Path, show_legend: bool) -> None:
        matplotlib.rcParams["hatch.linewidth"] = 1.5
        fig, ax = plt.subplots(figsize=(11.5, 7.0))
        for index, (_, label) in enumerate(METHOD_SPECS):
            offset = (index - (len(METHOD_SPECS) - 1) / 2.0) * width
            style = METHOD_STYLES[label]
            ax.bar(
                x + offset,
                plot_data[label],
                width=width,
                label=label,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="#333333",
                linewidth=0.9,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(architecture_labels, fontsize=20)
        ax.set_xlabel("Architecture", fontsize=22)
        ax.set_ylabel("Test loss barrier", fontsize=22)
        ax.tick_params(axis="both", labelsize=18, width=1.2, length=5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        if show_legend:
            ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=2,
                fontsize=16,
                frameon=True,
                framealpha=0.95,
                edgecolor="0.75",
                columnspacing=1.5,
                handlelength=2.4,
            )

        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    result_with_legend = output_root / "vgg_cifar10_three_way_test_loss_barriers.png"
    result_no_legend = output_root / "vgg_cifar10_three_way_test_loss_barriers_no_legend.png"
    save_barplot(result_with_legend, show_legend=True)
    save_barplot(result_no_legend, show_legend=False)

    (thesis_output_root / "barplot_vggs.png").write_bytes(
        result_with_legend.read_bytes()
    )
    (thesis_output_root / "barplot_vggs_no_legend.png").write_bytes(
        result_no_legend.read_bytes()
    )
    paper_output = (
        PROJECT_ROOT
        / "weekly_thesis_update(4)"
        / "paper_aistats2027"
        / "figures"
        / "barplot_vggs.png"
    )
    paper_output.write_bytes(result_with_legend.read_bytes())

    with open(output_root / "vgg_cifar10_three_way_test_loss_barriers.json", "w") as handle:
        json.dump(
            {
                "architectures": architecture_labels,
                "barriers": plot_data,
                "styles": METHOD_STYLES,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
