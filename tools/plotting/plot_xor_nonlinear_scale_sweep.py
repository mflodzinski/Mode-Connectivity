"""Aggregate the raw-endpoint XOR nonlinear-path scale sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


WIDTHS = (2, 3, 5, 7)
FAMILIES = ("polygonal", "bezier")
METHODS = ("path_only", "joint")
COLORS = {"path_only": "#0072B2", "joint": "#CC79A7"}
LABELS = {"path_only": "Nonlinear path", "joint": "Nonlinear path + scale"}


def result_path(root: Path, width: int, family: str) -> Path:
    return (
        root
        / f"xor_{width}h_joint_scale_{family}_sweep"
        / f"joint_scale_{family}_results.json"
    )


def load_results(root: Path) -> dict[tuple[int, str], dict]:
    results = {}
    for width in WIDTHS:
        for family in FAMILIES:
            path = result_path(root, width, family)
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            protocol = payload["config"].get("permutation_protocol")
            if protocol != "none; paths use raw trained endpoints":
                raise ValueError(f"Unexpected protocol in {path}: {protocol}")
            results[(width, family)] = payload
    return results


def barriers(bend_result: dict, method: str) -> np.ndarray:
    return np.asarray(
        [
            trial["methods"][method]["best"]["loss_barrier"]
            for pair in bend_result["pair_results"]
            for trial in pair["permutations"]
        ],
        dtype=np.float64,
    )


def successes(bend_result: dict, method: str) -> np.ndarray:
    return np.asarray(
        [
            trial["methods"][method]["best"]["low_loss_success"]
            for pair in bend_result["pair_results"]
            for trial in pair["permutations"]
        ],
        dtype=bool,
    )


def summarize(results: dict[tuple[int, str], dict]) -> list[dict]:
    rows = []
    for (width, family), payload in sorted(results.items()):
        for bend_result in payload["bend_results"]:
            internal_points = int(bend_result["num_internal_bends"])
            values = {method: barriers(bend_result, method) for method in METHODS}
            for method in METHODS:
                method_values = values[method]
                rows.append(
                    {
                        "width": width,
                        "family": family,
                        "internal_points": internal_points,
                        "method": method,
                        "num_pairs": len(method_values),
                        "mean_barrier": float(np.mean(method_values)),
                        "median_barrier": float(np.median(method_values)),
                        "q25_barrier": float(np.percentile(method_values, 25)),
                        "q75_barrier": float(np.percentile(method_values, 75)),
                        "low_loss_rate": float(
                            np.mean(successes(bend_result, method))
                        ),
                    }
                )
            delta = values["joint"] - values["path_only"]
            tolerance = 1e-8
            rows.append(
                {
                    "width": width,
                    "family": family,
                    "internal_points": internal_points,
                    "method": "paired_delta",
                    "num_pairs": len(delta),
                    "mean_barrier": float(np.mean(delta)),
                    "median_barrier": float(np.median(delta)),
                    "q25_barrier": float(np.percentile(delta, 25)),
                    "q75_barrier": float(np.percentile(delta, 75)),
                    "low_loss_rate": "",
                    "joint_better": int(np.sum(delta < -tolerance)),
                    "path_only_better": int(np.sum(delta > tolerance)),
                    "tied": int(np.sum(np.abs(delta) <= tolerance)),
                }
            )
    return rows


def write_summary(rows: list[dict], output: Path) -> None:
    fields = [
        "width",
        "family",
        "internal_points",
        "method",
        "num_pairs",
        "mean_barrier",
        "median_barrier",
        "q25_barrier",
        "q75_barrier",
        "low_loss_rate",
        "joint_better",
        "path_only_better",
        "tied",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_barriers(results: dict[tuple[int, str], dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.8), sharex=True, sharey=True)
    for row, family in enumerate(FAMILIES):
        for column, width in enumerate(WIDTHS):
            axis = axes[row, column]
            payload = results[(width, family)]
            x = np.asarray(
                [item["num_internal_bends"] for item in payload["bend_results"]]
            )
            for method in METHODS:
                samples = [barriers(item, method) for item in payload["bend_results"]]
                medians = np.asarray([np.median(values) for values in samples])
                lower = np.asarray([np.percentile(values, 25) for values in samples])
                upper = np.asarray([np.percentile(values, 75) for values in samples])
                axis.fill_between(x, lower, upper, color=COLORS[method], alpha=0.16)
                axis.plot(
                    x,
                    medians,
                    marker="o",
                    linewidth=2.1,
                    color=COLORS[method],
                    label=LABELS[method],
                )
            axis.set_yscale("symlog", linthresh=1e-5, linscale=0.8)
            axis.set_title(
                f"{family.capitalize()}, width {width}",
                fontsize=11.5,
                fontweight="bold",
            )
            axis.set_xticks(x)
            axis.grid(alpha=0.22)
            axis.tick_params(labelsize=9.5)
            if row == 1:
                axis.set_xlabel("Interior control points", fontsize=10.5)
            if column == 0:
                axis.set_ylabel("Loss barrier (symlog)", fontsize=10.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11.5,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_deltas(results: dict[tuple[int, str], dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.8), sharex=True)
    for row, family in enumerate(FAMILIES):
        for column, width in enumerate(WIDTHS):
            axis = axes[row, column]
            payload = results[(width, family)]
            x = np.asarray(
                [item["num_internal_bends"] for item in payload["bend_results"]]
            )
            samples = [
                barriers(item, "joint") - barriers(item, "path_only")
                for item in payload["bend_results"]
            ]
            medians = np.asarray([np.median(values) for values in samples])
            lower = np.asarray([np.percentile(values, 25) for values in samples])
            upper = np.asarray([np.percentile(values, 75) for values in samples])
            axis.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
            axis.fill_between(x, lower, upper, color="#009E73", alpha=0.18)
            axis.plot(x, medians, marker="o", linewidth=2.1, color="#009E73")
            axis.set_title(
                f"{family.capitalize()}, width {width}",
                fontsize=11.5,
                fontweight="bold",
            )
            axis.set_xticks(x)
            axis.grid(alpha=0.22)
            axis.tick_params(labelsize=9.5)
            if row == 1:
                axis.set_xlabel("Interior control points", fontsize=10.5)
            if column == 0:
                axis.set_ylabel(r"Barrier change, scale $-$ path", fontsize=10.5)
    fig.suptitle("Negative values favor scale", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_connectivity(results: dict[tuple[int, str], dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(14.2, 6.8), sharex=True, sharey=True)
    for row, family in enumerate(FAMILIES):
        for column, width in enumerate(WIDTHS):
            axis = axes[row, column]
            payload = results[(width, family)]
            x = np.asarray(
                [item["num_internal_bends"] for item in payload["bend_results"]]
            )
            for method in METHODS:
                rates = np.asarray(
                    [np.mean(successes(item, method)) for item in payload["bend_results"]]
                )
                offset = -0.035 if method == "path_only" else 0.035
                axis.plot(
                    x + offset,
                    rates,
                    marker="o" if method == "path_only" else "s",
                    linewidth=2.1,
                    linestyle="-" if method == "path_only" else "--",
                    color=COLORS[method],
                    label=LABELS[method],
                )
            axis.set_title(
                f"{family.capitalize()}, width {width}",
                fontsize=11.5,
                fontweight="bold",
            )
            axis.set_xticks(x)
            axis.set_ylim(-0.03, 1.03)
            axis.set_yticks(np.linspace(0.0, 1.0, 5))
            axis.grid(alpha=0.22)
            axis.tick_params(labelsize=9.5)
            if row == 1:
                axis.set_xlabel("Interior control points", fontsize=10.5)
            if column == 0:
                axis.set_ylabel("Low-loss connection rate", fontsize=10.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=11.5,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/xor"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/xor/nonlinear_scale_sweep_summary"),
    )
    args = parser.parse_args()

    results = load_results(args.root)
    rows = summarize(results)
    write_summary(rows, args.output_dir / "summary.csv")
    plot_connectivity(results, args.output_dir / "connectivity_rate.png")
    plot_barriers(results, args.output_dir / "barriers_by_control_points.png")
    plot_deltas(results, args.output_dir / "paired_barrier_change.png")
    print(f"Wrote sweep summary to {args.output_dir}")


if __name__ == "__main__":
    main()
