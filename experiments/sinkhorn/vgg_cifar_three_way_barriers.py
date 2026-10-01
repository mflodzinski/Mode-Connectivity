"""Aggregate three-way interpolation barriers across retained VGG architectures.

The script reads per-architecture comparison artifacts and reduces them into
the barrier plots used to summarize no-alignment, permutation-only, and
scale-aware alignment performance.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from mode_connectivity.common.paths import PROJECT_ROOT

from mode_connectivity.alignment.permutation_pipeline import compute_paper_loss_barrier


METHOD_LABELS = [
    "No alignment",
    "Permutation only (weight matching)",
    "Weight matching + scale refinement",
    "Permutation only (Sinkhorn)",
    "Permutation + scale (joint Sinkhorn)",
    "Sinkhorn + scale refinement",
]

VGG_CURVE_METHODS = {
    "No alignment": "test_naive",
    "Permutation only (Sinkhorn)": "test_perm",
    "Permutation + scale (joint Sinkhorn)": "test_scale",
}

ARCHITECTURES = ["vgg11", "vgg13", "vgg16", "vgg19"]

VGG11_REPORT_ROOT = PROJECT_ROOT / "results/dense_linear_stage_vgg11_10k/report"
FASHION_REPORT_ROOT = (
    PROJECT_ROOT / "results/dense_linear_stage_fashion_mnist_10k_atol2e4/report"
)
DENSE_METHODS = {
    "No alignment": "raw",
    "Permutation only (weight matching)": "wm",
    "Weight matching + scale refinement": "wm_scale",
    "Permutation only (Sinkhorn)": "sinkhorn",
    "Permutation + scale (joint Sinkhorn)": "sinkhorn_scale_joint",
    "Sinkhorn + scale refinement": "sinkhorn_scale_finetune",
}

# Match the Okabe--Ito-derived method colors used by the XOR profiles in
# Figure 2.  Hatching supplies a second, grayscale-safe encoding for
# scale-aware variants, while raw and permutation-only bars are solid.
METHOD_STYLES = {
    "No alignment": {"color": "#7A7A7A", "hatch": ""},
    "Permutation only (weight matching)": {"color": "#0072B2", "hatch": ""},
    "Weight matching + scale refinement": {
        "color": "#56B4E9",
        "hatch": "\\\\\\",
    },
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


def load_dense_final_barriers(report_root: Path) -> dict[str, float]:
    """Load seed-pair-0 final--final test-loss barriers.

    The dense experiment evaluates all six methods only for its calibration
    pair.  The coarse profiles are used for every method so all six values share the
    same 21-point interpolation grid.
    """

    summary_path = report_root / "summary.json"
    barriers_path = report_root / "barriers.csv"
    if not summary_path.exists() or not barriers_path.exists():
        raise FileNotFoundError(
            f"Missing completed dense-stage report under {report_root}"
        )
    summary = json.loads(summary_path.read_text())
    final_epoch = int(summary["stages"][-1])
    wanted = set(DENSE_METHODS.values())
    values: dict[str, float] = {}
    with barriers_path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if (
                int(row["replicate"]) == 0
                and int(row["left_epoch"]) == final_epoch
                and int(row["right_epoch"]) == final_epoch
                and row["method"] in wanted
                and row["resolution"] == "coarse"
                and row["subset"] == "test_full"
                and row["metric"] == "loss"
                and row["barrier"] == "chord"
            ):
                values[row["method"]] = float(row["value"])
    missing = wanted - set(values)
    if missing:
        raise ValueError(f"Missing dense-stage final barriers for {sorted(missing)}")
    return {label: values[method] for label, method in DENSE_METHODS.items()}


def main() -> None:
    plot_data: dict[str, list[float]] = {label: [] for label in METHOD_LABELS}
    architecture_labels: list[str] = []

    for architecture in ARCHITECTURES:
        payload = load_curves(architecture)
        architecture_labels.append(str(payload["vgg_name"]))
        if architecture == "vgg11":
            vgg11 = load_dense_final_barriers(VGG11_REPORT_ROOT)
            for label in METHOD_LABELS:
                plot_data[label].append(vgg11[label])
            continue

        curves = payload["curves"]
        for label in METHOD_LABELS:
            if label in VGG_CURVE_METHODS:
                curve = curves[VGG_CURVE_METHODS[label]]
                plot_data[label].append(
                    compute_barrier(curve["losses"], curve["lambdas"])
                )
            elif label == "Sinkhorn + scale refinement":
                plot_data[label].append(load_perm_then_scale_barrier(architecture))
            else:
                plot_data[label].append(float("nan"))

    fashion = load_dense_final_barriers(FASHION_REPORT_ROOT)
    architecture_labels.append("MLP-10×512\nFashion-MNIST")
    for label in METHOD_LABELS:
        plot_data[label].append(fashion[label])

    output_root = PROJECT_ROOT / "results" / "vgg_cifar10_three_way_barriers"
    output_root.mkdir(parents=True, exist_ok=True)
    thesis_output_root = PROJECT_ROOT / "thesis" / "figures" / "new"
    thesis_output_root.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(architecture_labels))
    width = 0.115

    def save_barplot(output_path: Path, show_legend: bool) -> None:
        matplotlib.rcParams["hatch.linewidth"] = 1.5
        fig, ax = plt.subplots(figsize=(12.5, 7.0))
        for label in METHOD_LABELS:
            style = METHOD_STYLES[label]
            positions = []
            values = []
            for architecture_index in range(len(architecture_labels)):
                available = [
                    candidate
                    for candidate in METHOD_LABELS
                    if np.isfinite(plot_data[candidate][architecture_index])
                ]
                if label not in available:
                    continue
                method_index = available.index(label)
                offset = (method_index - (len(available) - 1) / 2.0) * width
                positions.append(x[architecture_index] + offset)
                values.append(plot_data[label][architecture_index])
            ax.bar(
                positions,
                values,
                width=width,
                label=label,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="#333333",
                linewidth=0.9,
            )
            for position, value in zip(positions, values):
                if np.isclose(value, 0.0, atol=5e-5):
                    ax.annotate(
                        "0.000",
                        xy=(position, 0.0),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=11,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(architecture_labels, fontsize=17)
        ax.set_xlabel("Model and dataset", fontsize=22)
        ax.set_ylabel("Test loss barrier", fontsize=22)
        ax.tick_params(axis="both", labelsize=18, width=1.2, length=5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.8, alpha=0.45)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        if show_legend:
            handles, labels = ax.get_legend_handles_labels()
            legend_order = [0, 3, 1, 4, 2, 5]
            ax.legend(
                [handles[index] for index in legend_order],
                [labels[index] for index in legend_order],
                loc="lower center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=3,
                fontsize=14,
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
                "barriers": {
                    label: [
                        value if np.isfinite(value) else None for value in values
                    ]
                    for label, values in plot_data.items()
                },
                "styles": METHOD_STYLES,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
