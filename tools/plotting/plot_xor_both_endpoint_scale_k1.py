"""Plot the one-control XOR experiment with independent scaling at both endpoints."""

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
LABELS = {
    "path_only": "Nonlinear path",
    "joint": "Scale-aware nonlinear path",
}


def load_results(root: Path) -> dict[tuple[int, str], dict]:
    results = {}
    for width in WIDTHS:
        for family in FAMILIES:
            path = (
                root
                / f"xor_{width}h_joint_scale_{family}_both_endpoints_k1"
                / f"joint_scale_{family}_results.json"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            payload = json.loads(path.read_text())
            config = payload["config"]
            expected = {
                "permutation_protocol": "none; paths use raw trained endpoints",
                "scale_endpoints": "both",
                "internal_control_parameterization": "absolute",
                "num_internal_bends": [1],
            }
            for key, value in expected.items():
                if config.get(key) != value:
                    raise ValueError(
                        f"Unexpected {key} in {path}: {config.get(key)!r}"
                    )
            results[(width, family)] = payload["bend_results"][0]
    return results


def best_records(result: dict, method: str) -> list[dict]:
    return [
        trial["methods"][method]["best"]
        for pair in result["pair_results"]
        for trial in pair["permutations"]
    ]


def summarize(results: dict[tuple[int, str], dict]) -> list[dict]:
    rows = []
    for (width, family), result in sorted(results.items()):
        records = {method: best_records(result, method) for method in METHODS}
        for method, method_records in records.items():
            barriers = np.asarray([record["loss_barrier"] for record in method_records])
            rows.append(
                {
                    "width": width,
                    "family": family,
                    "method": method,
                    "num_pairs": len(barriers),
                    "num_low_loss": sum(
                        bool(record["low_loss_success"]) for record in method_records
                    ),
                    "low_loss_rate": float(
                        np.mean([record["low_loss_success"] for record in method_records])
                    ),
                    "mean_barrier": float(np.mean(barriers)),
                    "median_barrier": float(np.median(barriers)),
                }
            )
        path = np.asarray([record["loss_barrier"] for record in records["path_only"]])
        joint = np.asarray([record["loss_barrier"] for record in records["joint"]])
        delta = joint - path
        rows.append(
            {
                "width": width,
                "family": family,
                "method": "paired_delta",
                "num_pairs": len(delta),
                "num_low_loss": "",
                "low_loss_rate": "",
                "mean_barrier": float(np.mean(delta)),
                "median_barrier": float(np.median(delta)),
                "joint_better": int(np.sum(delta < -1e-8)),
                "path_only_better": int(np.sum(delta > 1e-8)),
                "tied": int(np.sum(np.abs(delta) <= 1e-8)),
            }
        )
    return rows


def write_summary(rows: list[dict], output: Path) -> None:
    fields = [
        "width",
        "family",
        "method",
        "num_pairs",
        "num_low_loss",
        "low_loss_rate",
        "mean_barrier",
        "median_barrier",
        "joint_better",
        "path_only_better",
        "tied",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_connectivity(results: dict[tuple[int, str], dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(WIDTHS), dtype=np.float64)
    conditions = [
        ("polygonal", "path_only", -0.27, ""),
        ("polygonal", "joint", -0.09, ""),
        ("bezier", "path_only", 0.09, "//"),
        ("bezier", "joint", 0.27, "//"),
    ]
    for family, method, offset, hatch in conditions:
        rates = [
            np.mean(
                [record["low_loss_success"] for record in best_records(results[(w, family)], method)]
            )
            for w in WIDTHS
        ]
        axis.bar(
            x + offset,
            rates,
            width=0.18,
            color=COLORS[method],
            edgecolor="white" if not hatch else "#444444",
            linewidth=0.7,
            hatch=hatch,
            label=f"{family.capitalize()}: {LABELS[method]}",
        )
    axis.set(
        xticks=x,
        xticklabels=[str(width) for width in WIDTHS],
        xlabel="Hidden width",
        ylabel="Low-loss connection rate",
        ylim=(0.0, 1.05),
    )
    axis.grid(axis="y", alpha=0.22)
    axis.legend(frameon=False, ncol=2, fontsize=9.5, loc="upper left")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_paired_barriers(results: dict[tuple[int, str], dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13.8, 6.8))
    for row, family in enumerate(FAMILIES):
        for column, width in enumerate(WIDTHS):
            axis = axes[row, column]
            result = results[(width, family)]
            path = np.asarray(
                [record["loss_barrier"] for record in best_records(result, "path_only")]
            )
            joint = np.asarray(
                [record["loss_barrier"] for record in best_records(result, "joint")]
            )
            limit = max(float(path.max()), float(joint.max()), 1e-5) * 1.08
            axis.scatter(
                path,
                joint,
                color="#CC79A7",
                edgecolor="white",
                linewidth=0.5,
                s=34,
                alpha=0.85,
            )
            axis.plot([0.0, limit], [0.0, limit], "--", color="#777777", lw=1.0)
            axis.set_xscale("symlog", linthresh=1e-5)
            axis.set_yscale("symlog", linthresh=1e-5)
            axis.set_xlim(-1e-6, limit)
            axis.set_ylim(-1e-6, limit)
            axis.set_title(
                f"{family.capitalize()}, width {width}",
                fontweight="bold",
                fontsize=11,
            )
            axis.grid(alpha=0.2)
            if row == 1:
                axis.set_xlabel("Path-only barrier")
            if column == 0:
                axis.set_ylabel("Path + endpoint scales barrier")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=250, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_mean_barriers(results: dict[tuple[int, str], dict], output: Path) -> None:
    x = np.arange(len(WIDTHS), dtype=np.float64)
    family_means = {}
    all_means = []
    for family in FAMILIES:
        means = {}
        for method in METHODS:
            means[method] = np.asarray(
                [
                    np.mean(
                        [
                            record["loss_barrier"]
                            for record in best_records(results[(width, family)], method)
                        ]
                    )
                    for width in WIDTHS
                ]
            )
            all_means.extend(means[method].tolist())
        family_means[family] = means

    output.parent.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        fig, axis = plt.subplots(figsize=(6.0, 5.6))
        means = family_means[family]
        for method in METHODS:
            axis.plot(
                x,
                means[method],
                marker="o" if method == "path_only" else "s",
                linestyle="-" if method == "path_only" else "--",
                linewidth=2.2,
                markersize=7,
                color=COLORS[method],
                label=LABELS[method],
            )
        annotation_y_offsets = (12, 12, 12, 18)
        for index, (baseline, scaled) in enumerate(
            zip(means["path_only"], means["joint"])
        ):
            percent = 100.0 * (scaled - baseline) / baseline
            axis.annotate(
                f"{percent:+.1f}%",
                xy=(x[index], max(baseline, scaled)),
                xytext=(0, annotation_y_offsets[index]),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=13,
                color="#444444",
            )
        axis.set_yscale("log")
        axis.set_xticks(x, [str(width) for width in WIDTHS])
        axis.set_xlim(-0.30, len(WIDTHS) - 1 + 0.30)
        axis.set_xlabel("Hidden width", fontsize=15)
        axis.set_title(
            "Endpoint scaling in nonlinear path finding",
            fontsize=13,
            fontweight="bold",
        )
        axis.grid(alpha=0.22, which="both")
        axis.tick_params(labelsize=10)
        axis.set_ylim(min(all_means) * 0.45, max(all_means) * 4.0)
        axis.set_ylabel("Mean loss barrier", fontsize=15)
        axis.legend(
            loc="lower left",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="#dddddd",
            fontsize=10,
        )
        fig.tight_layout()
        family_output = output.with_name(
            f"{output.stem}_{family}{output.suffix}"
        )
        fig.savefig(family_output, dpi=250, bbox_inches="tight")
        fig.savefig(family_output.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def plot_paired_change_with_std(
    results: dict[tuple[int, str], dict], output: Path
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.8, 3.9), sharey=True)
    family_colors = {"polygonal": "#009E73", "bezier": "#E69F00"}
    for axis, width in zip(axes, WIDTHS):
        for family_index, family in enumerate(FAMILIES):
            result = results[(width, family)]
            path = np.asarray(
                [record["loss_barrier"] for record in best_records(result, "path_only")]
            )
            joint = np.asarray(
                [record["loss_barrier"] for record in best_records(result, "joint")]
            )
            delta = joint - path
            jitter = np.linspace(-0.09, 0.09, len(delta))
            axis.scatter(
                family_index + jitter,
                delta,
                s=18,
                alpha=0.38,
                color=family_colors[family],
                edgecolor="none",
            )
            mean = float(np.mean(delta))
            standard_deviation = float(np.std(delta, ddof=1))
            axis.errorbar(
                family_index,
                mean,
                yerr=standard_deviation,
                fmt="D",
                markersize=6,
                capsize=5,
                elinewidth=1.8,
                color=family_colors[family],
                markeredgecolor="white",
                markeredgewidth=0.7,
                zorder=5,
            )
        axis.axhline(0.0, color="#777777", linestyle="--", linewidth=1.0)
        axis.set_yscale("symlog", linthresh=1e-5, linscale=0.8)
        axis.set_xticks((0, 1), ("Polygonal", "Bézier"))
        axis.set_title(f"Width {width}", fontweight="bold", fontsize=12)
        axis.grid(axis="y", alpha=0.22, which="both")
        axis.tick_params(axis="x", labelsize=9.5)
        axis.text(
            0.5,
            0.04,
            f"$n={len(best_records(results[(width, 'polygonal')], 'path_only'))}$ pairs",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#555555",
        )
    axes[0].set_ylabel(
        r"Paired barrier change $B_{\mathrm{scale}}-B_{\mathrm{path}}$",
        fontsize=10.5,
    )
    fig.suptitle(
        "Individual pairs and mean $\pm$ one standard deviation; negative favors scale",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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
        default=Path("results/xor/nonlinear_both_endpoints_k1_summary"),
    )
    args = parser.parse_args()
    results = load_results(args.root)
    rows = summarize(results)
    write_summary(rows, args.output_dir / "summary.csv")
    plot_connectivity(results, args.output_dir / "connectivity_rate.png")
    plot_mean_barriers(results, args.output_dir / "mean_barrier_by_width.png")
    plot_paired_change_with_std(
        results, args.output_dir / "paired_barrier_change_with_std.png"
    )
    plot_paired_barriers(results, args.output_dir / "paired_barriers.png")
    print(f"Wrote both-endpoint scale summary to {args.output_dir}")


if __name__ == "__main__":
    main()
