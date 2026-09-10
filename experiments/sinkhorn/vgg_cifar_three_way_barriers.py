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
    ("test_naive", "No Alignment"),
    ("test_perm", "Permutation Only"),
    ("test_scale", "Permutation + Scale"),
    ("perm_then_scale_only", "Permutation then Scale"),
]

ARCHITECTURES = ["vgg11", "vgg13", "vgg16", "vgg19"]

COLOR_CONFIGS = [
    (
        "color_config_1",
        {
            "No Alignment": "#BDBDBD",
            "Permutation Only": "#4C78A8",
            "Permutation + Scale": "#8E63CE",
            "Permutation then Scale": "#59A14F",
        },
    ),
    (
        "color_config_2",
        {
            "No Alignment": "#999999",
            "Permutation Only": "#377EB8",
            "Permutation + Scale": "#984EA3",
            "Permutation then Scale": "#4DAF4A",
        },
    ),
    (
        "color_config_3",
        {
            "No Alignment": "#CFCFCF",
            "Permutation Only": "#7AA6DC",
            "Permutation + Scale": "#B39DDB",
            "Permutation then Scale": "#81C784",
        },
    ),
]

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
        plot_data["Permutation then Scale"].append(load_perm_then_scale_barrier(architecture))

    output_root = PROJECT_ROOT / "results" / "vgg_cifar10_three_way_barriers"
    output_root.mkdir(parents=True, exist_ok=True)
    thesis_output_root = PROJECT_ROOT / "thesis" / "figures" / "new"
    thesis_output_root.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(architecture_labels))
    width = 0.14

    def save_barplot(output_path: Path, show_legend: bool, colors: dict[str, str]) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        for index, (_, label) in enumerate(METHOD_SPECS):
            offset = (index - (len(METHOD_SPECS) - 1) / 2.0) * width
            ax.bar(x + offset, plot_data[label], width=width, label=label, color=colors[label])

        ax.set_xticks(x)
        ax.set_xticklabels(architecture_labels, fontsize=16)
        ax.set_xlabel("Architecture", fontsize=18, fontweight="bold")
        ax.set_ylabel("Test Loss Barrier", fontsize=18, fontweight="bold")
        ax.tick_params(axis="y", labelsize=14)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, which="major", axis="both", linestyle="--", linewidth=0.7, alpha=0.5)
        ax.set_axisbelow(True)
        if show_legend:
            ax.legend(fontsize=13)

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    for index, (config_name, colors) in enumerate(COLOR_CONFIGS, start=1):
        result_stem = f"vgg_cifar10_three_way_test_loss_barriers_{config_name}"
        thesis_stem = f"barplot_vggs_{config_name}"

        result_with_legend = output_root / f"{result_stem}.png"
        result_no_legend = output_root / f"{result_stem}_no_legend.png"
        save_barplot(result_with_legend, show_legend=True, colors=colors)
        save_barplot(result_no_legend, show_legend=False, colors=colors)

        thesis_with_legend = thesis_output_root / f"{thesis_stem}.png"
        thesis_no_legend = thesis_output_root / f"{thesis_stem}_no_legend.png"
        thesis_with_legend.write_bytes(result_with_legend.read_bytes())
        thesis_no_legend.write_bytes(result_no_legend.read_bytes())

        if index == 1:
            canonical_with_legend = output_root / "vgg_cifar10_three_way_test_loss_barriers.png"
            canonical_no_legend = output_root / "vgg_cifar10_three_way_test_loss_barriers_no_legend.png"
            canonical_with_legend.write_bytes(result_with_legend.read_bytes())
            canonical_no_legend.write_bytes(result_no_legend.read_bytes())
            (thesis_output_root / "barplot_vggs.png").write_bytes(result_with_legend.read_bytes())
            (thesis_output_root / "barplot_vggs_no_legend.png").write_bytes(result_no_legend.read_bytes())

    with open(output_root / "vgg_cifar10_three_way_test_loss_barriers.json", "w") as handle:
        json.dump(
            {
                "architectures": architecture_labels,
                "barriers": plot_data,
                "color_configs": [
                    {
                        "name": config_name,
                        "colors": colors,
                    }
                    for config_name, colors in COLOR_CONFIGS
                ],
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
