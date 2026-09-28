"""Plot nonlinear XOR paths against their jointly scale-aware counterparts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {
    "path_only": "#0072B2",
    "joint": "#CC79A7",
}
LABELS = {
    "path_only": "Nonlinear path",
    "joint": "Nonlinear path + scale",
}


def load_width_result(path: Path, bend_count: int) -> dict:
    payload = json.loads(path.read_text())
    for result in payload["bend_results"]:
        if int(result["num_internal_bends"]) == bend_count:
            return result
    available = [result["num_internal_bends"] for result in payload["bend_results"]]
    raise ValueError(f"Bend count {bend_count} absent from {path}; available: {available}")


def best_curves(bend_result: dict, method: str) -> tuple[np.ndarray, np.ndarray]:
    trials = [
        trial
        for pair in bend_result["pair_results"]
        for trial in pair["permutations"]
    ]
    curves = np.asarray(
        [trial["methods"][method]["best"]["loss"] for trial in trials],
        dtype=np.float64,
    )
    locations = np.asarray(
        trials[0]["methods"][method]["best"]["t"], dtype=np.float64
    )
    return locations, curves


def parse_result(value: str) -> tuple[int, Path]:
    width, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("Expected WIDTH=PATH")
    return int(width), Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result",
        action="append",
        type=parse_result,
        required=True,
        metavar="WIDTH=PATH",
    )
    parser.add_argument("--bend-count", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/xor/xor_nonlinear_path_vs_scale.png"),
    )
    args = parser.parse_args()

    width_results = sorted(args.result)
    fig, axes = plt.subplots(
        1, len(width_results), figsize=(4.4 * len(width_results), 4.0), squeeze=False
    )
    axes = list(axes[0])
    summaries = []

    for panel_index, ((width, path), axis) in enumerate(zip(width_results, axes)):
        bend_result = load_width_result(path, args.bend_count)
        for method in ("path_only", "joint"):
            locations, curves = best_curves(bend_result, method)
            for curve in curves:
                axis.plot(
                    locations,
                    curve,
                    color=COLORS[method],
                    alpha=0.08,
                    linewidth=0.7,
                )
            lower, median, upper = np.percentile(curves, [25, 50, 75], axis=0)
            axis.fill_between(
                locations, lower, upper, color=COLORS[method], alpha=0.18
            )
            axis.plot(
                locations,
                median,
                color=COLORS[method],
                linewidth=2.5,
                label=LABELS[method],
            )

        path_barriers = np.asarray(
            [
                trial["methods"]["path_only"]["best"]["loss_barrier"]
                for pair in bend_result["pair_results"]
                for trial in pair["permutations"]
            ]
        )
        joint_barriers = np.asarray(
            [
                trial["methods"]["joint"]["best"]["loss_barrier"]
                for pair in bend_result["pair_results"]
                for trial in pair["permutations"]
            ]
        )
        summaries.append(
            {
                "width": width,
                "pairs": len(path_barriers),
                "path_median_barrier": float(np.median(path_barriers)),
                "joint_median_barrier": float(np.median(joint_barriers)),
                "joint_better_pairs": int(np.sum(joint_barriers < path_barriers)),
            }
        )

        axis.set_title(f"Hidden size {width}", fontweight="bold", fontsize=14)
        axis.set_xlabel(r"$\lambda$", fontsize=13)
        if panel_index == 0:
            axis.set_ylabel("Binary cross-entropy", fontsize=13)
        axis.tick_params(axis="both", labelsize=11)
        axis.grid(alpha=0.22)

    legend_axis = axes[len(axes) // 2]
    legend_axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        frameon=False,
        fontsize=13,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=250, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps(summaries, indent=2))
    print(f"PNG: {args.output}")
    print(f"PDF: {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
