"""
Plot relative train-accuracy barrier vs L2 distance for the iteration-split
controlled benchmark, aggregating multiple weight-matching seeds per split.

For each split iteration:
- the original point (w0 <-> w1) is deterministic
- the recovered point (w0 <-> w1_recovered) is aggregated across wm_seed runs

Recovered points are plotted with mean +/- std error bars in both x and y.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

EPS = 1e-12


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def relative_barrier(accs, eps=EPS):
    endpoint_avg = 0.5 * (accs[0] + accs[-1])
    return (endpoint_avg - min(accs)) / (endpoint_avg + eps)


def absolute_barrier(accs):
    endpoint_avg = 0.5 * (accs[0] + accs[-1])
    return endpoint_avg - min(accs)


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot relative train-accuracy barrier vs L2 distance for iter-split benchmark aggregated over WM seeds."
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="results/analysis/alignment_benchmark_iter_noaug",
        help="Root with iter{split}/wm_seed{seed}/results.json",
    )
    parser.add_argument(
        "--splits",
        type=int,
        nargs="+",
        default=[0, 25, 100, 1000, 5000],
        help="Split iterations to include",
    )
    parser.add_argument(
        "--wm-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Weight-matching seeds to aggregate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/alignment_train_accuracy_iter_noaug_wm.png",
        help="Output file path",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="plots/alignment_train_accuracy_iter_noaug_wm.csv",
        help="Optional CSV summary output path",
    )
    parser.add_argument(
        "--barrier-mode",
        choices=["relative", "absolute"],
        default="relative",
        help="Whether to plot relative or absolute train-accuracy barrier.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["iterations", "epochs"],
        default="iterations",
        help="How to annotate split points on the plot.",
    )
    parser.add_argument(
        "--iters-per-epoch",
        type=float,
        default=390.0,
        help="Iterations per epoch used when --label-mode=epochs.",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = PROJECT_ROOT / args.results_root
    barrier_fn = relative_barrier if args.barrier_mode == "relative" else absolute_barrier

    experiments = []
    for split in args.splits:
        per_seed = []
        for wm_seed in args.wm_seeds:
            json_path = results_root / f"iter{split}" / f"wm_seed{wm_seed}" / "results.json"
            if not json_path.exists():
                print(f"Warning: missing {json_path}")
                continue
            per_seed.append(load_json(json_path))

        if not per_seed:
            print(f"Warning: no results found for split {split}")
            continue

        original = per_seed[0]
        rec_xs = [run["distances"]["w0_w1_recovered"]["l2_distance"] for run in per_seed]
        rec_ys = [barrier_fn(run["recovered_barrier_to_w0"]["train_acc"]) for run in per_seed]

        rec_x_mean, rec_x_std = mean_std(rec_xs)
        rec_y_mean, rec_y_std = mean_std(rec_ys)

        if args.label_mode == "epochs":
            display_name = f"{split / args.iters_per_epoch:.2f}"
        else:
            display_name = str(split)

        experiments.append(
            {
                "name": display_name,
                "split_iter": split,
                "w0_w1": original["distances"]["w0_w1"]["l2_distance"],
                "org_barrier": barrier_fn(original["original_barrier"]["train_acc"]),
                "rec_xs": rec_xs,
                "rec_ys": rec_ys,
                "w0_w1_recovered_mean": rec_x_mean,
                "w0_w1_recovered_std": rec_x_std,
                "rec_barrier_mean": rec_y_mean,
                "rec_barrier_std": rec_y_std,
                "num_runs": len(per_seed),
            }
        )

    if not experiments:
        raise RuntimeError("No benchmark results found.")

    experiments.sort(key=lambda e: e["split_iter"])

    fig, ax = plt.subplots(figsize=(10, 6))

    org_color = "#1f77b4"
    rec_color = "#2ca02c"

    x_org = [e["w0_w1"] for e in experiments]
    y_org = [e["org_barrier"] for e in experiments]
    ax.scatter(x_org, y_org, c=org_color, marker="o", s=100, label="Original (shared training)", zorder=5)

    first_seed_label = True
    for e in experiments:
        label = "Recovered WM seeds" if first_seed_label else None
        first_seed_label = False
        ax.scatter(
            e["rec_xs"],
            e["rec_ys"],
            c=rec_color,
            marker="o",
            s=28,
            alpha=0.35,
            linewidths=0,
            label=label,
            zorder=4,
        )

    x_rec = [e["w0_w1_recovered_mean"] for e in experiments]
    y_rec = [e["rec_barrier_mean"] for e in experiments]
    xerr = [e["w0_w1_recovered_std"] for e in experiments]
    yerr = [e["rec_barrier_std"] for e in experiments]

    ax.errorbar(
        x_rec,
        y_rec,
        xerr=xerr,
        yerr=yerr,
        fmt="o",
        color=rec_color,
        ecolor=rec_color,
        elinewidth=2.2,
        capsize=6,
        capthick=2.0,
        markersize=10,
        markerfacecolor="white",
        markeredgewidth=2.0,
        label="Recovered mean ± std (3 WM seeds)",
        zorder=7,
    )

    for e in experiments:
        ax.annotate(
            e["name"],
            (e["w0_w1"], e["org_barrier"]),
            textcoords="offset points",
            xytext=(-5, 8),
            fontsize=8,
            color=org_color,
        )
        ax.annotate(
            "",
            xy=(e["w0_w1_recovered_mean"], e["rec_barrier_mean"]),
            xytext=(e["w0_w1"], e["org_barrier"]),
            arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7, lw=1.5),
        )

    ax.set_xlabel("L2 Distance", fontsize=12)
    if args.barrier_mode == "relative":
        ax.set_ylabel("Relative Train-Accuracy Barrier", fontsize=12)
    else:
        ax.set_ylabel("Absolute Train-Accuracy Barrier", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    csv_path = PROJECT_ROOT / args.csv_output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split_iter",
                "w0_w1",
                "org_barrier",
                "w0_w1_recovered_mean",
                "w0_w1_recovered_std",
                "rec_barrier_mean",
                "rec_barrier_std",
                "num_runs",
            ],
        )
        writer.writeheader()
        for e in experiments:
            writer.writerow(
                {
                    "split_iter": e["split_iter"],
                    "w0_w1": e["w0_w1"],
                    "org_barrier": e["org_barrier"],
                    "w0_w1_recovered_mean": e["w0_w1_recovered_mean"],
                    "w0_w1_recovered_std": e["w0_w1_recovered_std"],
                    "rec_barrier_mean": e["rec_barrier_mean"],
                    "rec_barrier_std": e["rec_barrier_std"],
                    "num_runs": e["num_runs"],
                }
            )
    print(f"Saved CSV summary to {csv_path}")

    if args.show:
        plt.show()

    print("\n" + "=" * 120)
    print(
        f"ITER-SPLIT CONTROLLED BENCHMARK SUMMARY "
        f"(TRAIN {args.barrier_mode.upper()} BARRIER, AGGREGATED OVER WM SEEDS)"
    )
    print("=" * 120)
    print(
        f"{'split':>8} {'orig_x':>10} {'orig_bar':>12} {'rec_x_mean':>12} {'rec_x_std':>11} "
        f"{'rec_bar_mean':>13} {'rec_bar_std':>12} {'n':>4}"
    )
    print("-" * 120)
    for e in experiments:
        print(
            f"{e['split_iter']:>8d} "
            f"{e['w0_w1']:>10.2f} "
            f"{e['org_barrier']:>12.6f} "
            f"{e['w0_w1_recovered_mean']:>12.2f} "
            f"{e['w0_w1_recovered_std']:>11.4f} "
            f"{e['rec_barrier_mean']:>13.6f} "
            f"{e['rec_barrier_std']:>12.6f} "
            f"{e['num_runs']:>4d}"
        )


if __name__ == "__main__":
    main()
