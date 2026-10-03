"""Plot final-alignment barriers across architectures with replicate SDs.

Every value comes from the common final-alignment benchmark: three disjoint
endpoint pairs and pair-specific validation-only hyperparameter selection.
The script produces matching training-subset and official-test figures.
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


METHOD_LABELS = [
    "No alignment",
    "Permutation only (weight matching)",
    "Weight matching + scale refinement",
    "Permutation only (Sinkhorn)",
    "Permutation + scale (joint Sinkhorn)",
    "Sinkhorn + scale refinement",
]

DENSE_METHODS = {
    "No alignment": "raw",
    "Permutation only (weight matching)": "wm",
    "Weight matching + scale refinement": "wm_scale",
    "Permutation only (Sinkhorn)": "sinkhorn",
    "Permutation + scale (joint Sinkhorn)": "sinkhorn_scale_joint",
    "Sinkhorn + scale refinement": "sinkhorn_scale_finetune",
}

EXPERIMENTS = [
    (
        "VGG11\nCIFAR-10",
        PROJECT_ROOT / "results/final_alignment_vgg11_cifar10",
        200,
    ),
    (
        "VGG13\nCIFAR-10",
        PROJECT_ROOT / "results/final_alignment_vgg13_cifar10",
        200,
    ),
    (
        "VGG16\nCIFAR-10",
        PROJECT_ROOT / "results/final_alignment_vgg16_cifar10",
        200,
    ),
    (
        "VGG19\nCIFAR-10",
        PROJECT_ROOT
        / "results/final_alignment_vgg19_cifar10_atol5e5",
        200,
    ),
    (
        "MLP-10×512\nFashion-MNIST",
        PROJECT_ROOT / "results/final_alignment_fashion_mnist",
        100,
    ),
]

# Match the Okabe--Ito-derived method colors used by the XOR profiles.
# Hatching provides a second, grayscale-safe encoding for scale-aware methods.
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


def load_final_barriers(
    experiment_root: Path,
    epoch: int,
    subset: str,
) -> dict[str, dict]:
    """Load chord barriers from three pairs' interpolation profiles."""

    values_by_method = {method: [] for method in DENSE_METHODS.values()}
    for replicate in range(3):
        path = (
            experiment_root
            / "pairs"
            / f"r{replicate}"
            / f"{epoch:03d}_{epoch:03d}"
            / "full_profiles.json"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing final-alignment profile: {path}")
        profiles = json.loads(path.read_text())
        for method in values_by_method:
            key = f"{method}/{subset}/selected_dense"
            if key not in profiles:
                raise ValueError(f"Missing {key} in {path}")
            values_by_method[method].append(
                float(profiles[key]["loss"]["chord"])
            )

    result = {}
    for label, method in DENSE_METHODS.items():
        values = values_by_method[method]
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        if not np.isfinite([mean, std, *values]).all():
            raise FloatingPointError(
                f"Nonfinite aggregate for {method} in {experiment_root}"
            )
        result[label] = dict(mean=mean, std=std, values=values, count=3)
    return result


def load_full_train_barriers(experiment_root: Path) -> dict[str, dict]:
    """Load loss barriers evaluated on the complete endpoint-training split."""

    path = experiment_root / "report_full_train" / "aggregates.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing full-train aggregate report: {path}")
    rows = json.loads(path.read_text())
    by_method = {
        row["method"]: row
        for row in rows
        if row["metric"] == "loss" and row["subset"] == "train_fit"
    }
    result = {}
    for label, method in DENSE_METHODS.items():
        if method not in by_method:
            raise ValueError(f"Missing full-train loss aggregate for {method} in {path}")
        row = by_method[method]
        values = [float(value) for value in row["values"]]
        mean, std = float(row["mean"]), float(row["std"])
        if len(values) != 3 or not np.isfinite([mean, std, *values]).all():
            raise ValueError(f"Invalid full-train aggregate for {method} in {path}")
        result[label] = dict(mean=mean, std=std, values=values, count=3)
    return result


def main() -> None:
    architecture_labels = [label for label, _, _ in EXPERIMENTS]
    output_root = PROJECT_ROOT / "results" / "vgg_cifar10_three_way_barriers"
    output_root.mkdir(parents=True, exist_ok=True)
    thesis_output_root = PROJECT_ROOT / "thesis" / "figures" / "new"
    thesis_output_root.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(architecture_labels))
    width = 0.115

    def aggregate(subset: str) -> tuple[dict, dict, dict]:
        plot_means = {label: [] for label in METHOD_LABELS}
        plot_stds = {label: [] for label in METHOD_LABELS}
        replicate_values = {label: [] for label in METHOD_LABELS}
        for _, experiment_root, epoch in EXPERIMENTS:
            aggregates = load_final_barriers(experiment_root, epoch, subset)
            for label in METHOD_LABELS:
                plot_means[label].append(aggregates[label]["mean"])
                plot_stds[label].append(aggregates[label]["std"])
                replicate_values[label].append(aggregates[label]["values"])
        return plot_means, plot_stds, replicate_values

    def aggregate_full_train() -> tuple[dict, dict, dict]:
        plot_means = {label: [] for label in METHOD_LABELS}
        plot_stds = {label: [] for label in METHOD_LABELS}
        replicate_values = {label: [] for label in METHOD_LABELS}
        for _, experiment_root, _ in EXPERIMENTS:
            aggregates = load_full_train_barriers(experiment_root)
            for label in METHOD_LABELS:
                plot_means[label].append(aggregates[label]["mean"])
                plot_stds[label].append(aggregates[label]["std"])
                replicate_values[label].append(aggregates[label]["values"])
        return plot_means, plot_stds, replicate_values

    def save_barplot(
        output_path: Path,
        show_legend: bool,
        ylabel: str,
        plot_means: dict,
        plot_stds: dict,
    ) -> None:
        matplotlib.rcParams["hatch.linewidth"] = 1.5
        fig, ax = plt.subplots(figsize=(12.5, 7.0))
        for method_index, label in enumerate(METHOD_LABELS):
            style = METHOD_STYLES[label]
            positions = x + (method_index - (len(METHOD_LABELS) - 1) / 2.0) * width
            means = np.asarray(plot_means[label], dtype=float)
            stds = np.asarray(plot_stds[label], dtype=float)
            ax.bar(
                positions,
                means,
                yerr=stds,
                width=width,
                capsize=3,
                error_kw={"elinewidth": 1.0, "capthick": 1.0},
                label=label,
                color=style["color"],
                hatch=style["hatch"],
                edgecolor="#333333",
                linewidth=0.9,
            )
            for position, value in zip(positions, means):
                if np.isclose(value, 0.0, atol=5e-5):
                    ax.annotate(
                        "0.000",
                        xy=(position, 0.0),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        rotation=90,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(architecture_labels, fontsize=16)
        ax.set_xlabel("Model and dataset", fontsize=22)
        ax.set_ylabel(ylabel, fontsize=22)
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

    paper_figure_root = (
        PROJECT_ROOT
        / "weekly_thesis_update(4)"
        / "paper_aistats2027"
        / "figures"
    )
    figure_specs = (
        ("test_full", "test", "Test loss barrier", "barplot_vggs"),
        (
            "train_report",
            "train",
            "Training-subset loss barrier",
            "barplot_vggs_train_subset",
        ),
    )
    for subset, file_label, ylabel, paper_stem in figure_specs:
        plot_means, plot_stds, replicate_values = aggregate(subset)
        result_with_legend = (
            output_root / f"vgg_cifar10_three_way_{file_label}_loss_barriers.png"
        )
        result_no_legend = (
            output_root
            / f"vgg_cifar10_three_way_{file_label}_loss_barriers_no_legend.png"
        )
        save_barplot(
            result_with_legend,
            show_legend=True,
            ylabel=ylabel,
            plot_means=plot_means,
            plot_stds=plot_stds,
        )
        save_barplot(
            result_no_legend,
            show_legend=False,
            ylabel=ylabel,
            plot_means=plot_means,
            plot_stds=plot_stds,
        )

        (thesis_output_root / f"{paper_stem}.png").write_bytes(
            result_with_legend.read_bytes()
        )
        (thesis_output_root / f"{paper_stem}_no_legend.png").write_bytes(
            result_no_legend.read_bytes()
        )
        paper_output = paper_figure_root / f"{paper_stem}.png"
        paper_output.write_bytes(result_with_legend.read_bytes())

        payload = {
            "architectures": architecture_labels,
            "metric": f"{ylabel.lower()} above the endpoint chord",
            "subset": subset,
            "replicates": 3,
            "statistic": (
                "mean and sample standard deviation across disjoint endpoint pairs"
            ),
            "barriers": plot_means,
            "standard_deviations": plot_stds,
            "pair_values": replicate_values,
            "styles": METHOD_STYLES,
        }
        data_output = (
            output_root / f"vgg_cifar10_three_way_{file_label}_loss_barriers.json"
        )
        data_output.write_text(json.dumps(payload, indent=2) + "\n")
        paper_output.with_name(f"{paper_stem}_data.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    plot_means, plot_stds, replicate_values = aggregate_full_train()
    result_with_legend = (
        output_root / "vgg_cifar10_three_way_full_train_loss_barriers.png"
    )
    result_no_legend = (
        output_root / "vgg_cifar10_three_way_full_train_loss_barriers_no_legend.png"
    )
    save_barplot(
        result_with_legend,
        show_legend=True,
        ylabel="Train loss barrier",
        plot_means=plot_means,
        plot_stds=plot_stds,
    )
    save_barplot(
        result_no_legend,
        show_legend=False,
        ylabel="Train loss barrier",
        plot_means=plot_means,
        plot_stds=plot_stds,
    )
    paper_stem = "barplot_vggs_train"
    (thesis_output_root / f"{paper_stem}.png").write_bytes(
        result_with_legend.read_bytes()
    )
    (thesis_output_root / f"{paper_stem}_no_legend.png").write_bytes(
        result_no_legend.read_bytes()
    )
    paper_output = paper_figure_root / f"{paper_stem}.png"
    paper_output.write_bytes(result_with_legend.read_bytes())
    payload = {
        "architectures": architecture_labels,
        "metric": "training loss barrier above the endpoint chord",
        "subset": "train_fit",
        "subset_description": "complete endpoint-training split",
        "examples": [45000, 45000, 45000, 45000, 55000],
        "replicates": 3,
        "statistic": "mean and sample standard deviation across disjoint endpoint pairs",
        "barriers": plot_means,
        "standard_deviations": plot_stds,
        "pair_values": replicate_values,
        "styles": METHOD_STYLES,
    }
    data_output = (
        output_root / "vgg_cifar10_three_way_full_train_loss_barriers.json"
    )
    data_output.write_text(json.dumps(payload, indent=2) + "\n")
    paper_output.with_name(f"{paper_stem}_data.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
