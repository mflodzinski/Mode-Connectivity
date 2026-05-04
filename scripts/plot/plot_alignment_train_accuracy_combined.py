"""Combine the original controlled-benchmark train-accuracy plot with iter-noaug WM results.

Base layer:
- plots the original shared-split benchmark from plot_alignment_train_accuracy_v2.py

Overlay:
- plots the iteration-split no-augmentation WM aggregation results
- split labels are shown in epochs, using a user-provided iterations-per-epoch factor
"""

from __future__ import annotations

import argparse
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


def mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine epoch-split and iter-split controlled benchmark plots.")
    parser.add_argument(
        "--legacy-output",
        type=str,
        default="plots/alignment_train_accuracy_v2_with_iter_noaug_epochs.png",
        help="Output file path for combined figure.",
    )
    parser.add_argument(
        "--iter-results-root",
        type=str,
        default="results/analysis/alignment_benchmark_iter_noaug",
        help="Root with iter{split}/wm_seed{seed}/results.json",
    )
    parser.add_argument(
        "--iter-splits",
        type=int,
        nargs="+",
        default=[0, 25, 100, 1000, 5000],
        help="Iter-split points to include.",
    )
    parser.add_argument(
        "--wm-seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="WM seeds to aggregate for iter-split overlay.",
    )
    parser.add_argument(
        "--iters-per-epoch",
        type=float,
        default=390.0,
        help="Used to convert iteration split labels into epoch labels.",
    )
    parser.add_argument(
        "--x-mode",
        choices=["distance", "shared_epochs"],
        default="distance",
        help="Use either L2 distance or shared training epochs on the x-axis.",
    )
    parser.add_argument(
        "--iter-overlay-mode",
        choices=["all", "original_only"],
        default="all",
        help="Whether iter-noaug overlay includes both original and WM-recovered results or only original points.",
    )
    parser.add_argument(
        "--legacy-overlay-mode",
        choices=["all", "original_only"],
        default="all",
        help="Whether legacy benchmark includes both original and WM-recovered results or only original points.",
    )
    parser.add_argument("--show", action="store_true", help="Show interactively.")
    return parser.parse_args()


def load_legacy_experiments():
    data_files = {
        "150/50": "results/analysis/alignment_benchmark/results.json",
        "80/120": "results/analysis/alignment_benchmark_80split/results.json",
        "30/170": "results/analysis/alignment_benchmark_30split/results.json",
        "8/192": "results/analysis/alignment_benchmark_8split/results.json",
        "0/200": "results/analysis/alignment_benchmark_0split/results.json",
    }
    seed_file = "results/analysis/alignment_independent/seed0_seed1_results.json"

    experiments = []
    for name, rel_path in data_files.items():
        path = PROJECT_ROOT / rel_path
        if not path.exists():
            continue
        data = load_json(path)
        experiments.append(
            {
                "name": name,
                "w0_w1": data["distances"]["w0_w1"]["l2_distance"],
                "w0_w1_recovered": data["distances"]["w0_w1_recovered"]["l2_distance"],
                "org_rel_barrier": relative_barrier(data["original_barrier"]["train_acc"]),
                "rec_rel_barrier": relative_barrier(data["recovered_barrier_to_w0"]["train_acc"]),
                "is_independent": False,
            }
        )

    seed_path = PROJECT_ROOT / seed_file
    if seed_path.exists():
        data = load_json(seed_path)
        experiments.append(
            {
                "name": "seed0-seed1",
                "w0_w1": data["before_alignment"]["distance"]["l2_distance"],
                "w0_w1_recovered": data["after_alignment"]["distance"]["l2_distance"],
                "org_rel_barrier": relative_barrier(data["before_alignment"]["barrier"]["train_acc"]),
                "rec_rel_barrier": relative_barrier(data["after_alignment"]["barrier"]["train_acc"]),
                "is_independent": True,
                "show_original": True,
            }
        )
    return experiments


def legacy_name_to_shared_epochs(name: str) -> float:
    if "/" in name:
        return float(name.split("/")[0])
    raise ValueError(f"Cannot parse shared epochs from {name}")


def load_iter_overlay_experiments(results_root: Path, splits: list[int], wm_seeds: list[int], iters_per_epoch: float):
    experiments = []
    for split in splits:
        per_seed = []
        for wm_seed in wm_seeds:
            json_path = results_root / f"iter{split}" / f"wm_seed{wm_seed}" / "results.json"
            if json_path.exists():
                per_seed.append(load_json(json_path))
        if not per_seed:
            continue

        original = per_seed[0]
        rec_xs = [run["distances"]["w0_w1_recovered"]["l2_distance"] for run in per_seed]
        rec_ys = [relative_barrier(run["recovered_barrier_to_w0"]["train_acc"]) for run in per_seed]
        experiments.append(
            {
                "split_iter": split,
                "epoch_label": f"{split / iters_per_epoch:.2f}",
                "w0_w1": original["distances"]["w0_w1"]["l2_distance"],
                "org_rel_barrier": relative_barrier(original["original_barrier"]["train_acc"]),
                "rec_xs": rec_xs,
                "rec_ys": rec_ys,
                "rec_x_mean": mean_std(rec_xs)[0],
                "rec_x_std": mean_std(rec_xs)[1],
                "rec_y_mean": mean_std(rec_ys)[0],
                "rec_y_std": mean_std(rec_ys)[1],
            }
        )
    experiments.sort(key=lambda e: e["split_iter"])
    return experiments


def main() -> None:
    args = parse_args()
    legacy_experiments = load_legacy_experiments()
    iter_experiments = load_iter_overlay_experiments(
        PROJECT_ROOT / args.iter_results_root,
        args.iter_splits,
        args.wm_seeds,
        args.iters_per_epoch,
    )

    if not legacy_experiments:
        raise RuntimeError("No legacy benchmark data found.")
    if not iter_experiments:
        raise RuntimeError("No iter-split benchmark data found.")

    fig, ax = plt.subplots(figsize=(10, 6))

    org_color = "#1f77b4"
    rec_color = "#2ca02c"
    ind_diff_init_color = "#d62728"
    iter_org_color = "#4c78a8"
    iter_rec_color = "#ff7f0e"

    lmc_exps = [e for e in legacy_experiments if not e.get("is_independent", False)]
    ind_diff_init_exps = [e for e in legacy_experiments if e.get("is_independent", False)]

    x_org = [e["w0_w1"] for e in lmc_exps]
    y_org = [e["org_rel_barrier"] for e in lmc_exps]
    if args.x_mode == "shared_epochs":
        x_org = [legacy_name_to_shared_epochs(e["name"]) for e in lmc_exps]
    ax.scatter(x_org, y_org, c=org_color, marker="o", s=100, label="Original (shared training)", zorder=5)

    if args.legacy_overlay_mode == "all":
        x_rec = [e["w0_w1_recovered"] for e in lmc_exps]
        y_rec = [e["rec_rel_barrier"] for e in lmc_exps]
        if args.x_mode == "shared_epochs":
            x_rec = [legacy_name_to_shared_epochs(e["name"]) for e in lmc_exps]
        ax.scatter(x_rec, y_rec, c=rec_color, marker="o", s=100, label="Recovered (shared training)", zorder=5)
    else:
        x_rec = x_org

    for e, x0, x1 in zip(lmc_exps, x_org, x_rec):
        ax.annotate(
            e["name"],
            (x0, e["org_rel_barrier"]),
            textcoords="offset points",
            xytext=(-5, 8),
            fontsize=8,
            color=org_color,
        )
        if args.legacy_overlay_mode == "all":
            ax.annotate(
                "",
                xy=(x1, e["rec_rel_barrier"]),
                xytext=(x0, e["org_rel_barrier"]),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7, lw=1.5),
            )

    if args.legacy_overlay_mode == "all":
        first_diff_init = True
        first_diff_init_orig = True
        for e in ind_diff_init_exps:
            if e.get("show_original", False):
                x_orig = e["w0_w1"] if args.x_mode == "distance" else 200.0
                x_rec = e["w0_w1_recovered"] if args.x_mode == "distance" else 200.0
                label_orig = "Original (diff init)" if first_diff_init_orig else None
                first_diff_init_orig = False
                ax.scatter(
                    [x_orig],
                    [e["org_rel_barrier"]],
                    c=ind_diff_init_color,
                    marker="D",
                    s=150,
                    edgecolors="black",
                    linewidths=1.5,
                    alpha=0.5,
                    label=label_orig,
                    zorder=6,
                )
                ax.annotate(
                    "",
                    xy=(x_rec, e["rec_rel_barrier"]),
                    xytext=(x_orig, e["org_rel_barrier"]),
                    arrowprops=dict(arrowstyle="->", color=ind_diff_init_color, alpha=0.7, lw=2),
                )

            label_rec = "Recovered (diff init)" if first_diff_init else None
            first_diff_init = False
            x_rec = e["w0_w1_recovered"] if args.x_mode == "distance" else 200.0
            ax.scatter(
                [x_rec],
                [e["rec_rel_barrier"]],
                c=ind_diff_init_color,
                marker="D",
                s=150,
                edgecolors="black",
                linewidths=1.5,
                label=label_rec,
                zorder=6,
            )

    iter_org_x = [e["w0_w1"] for e in iter_experiments]
    iter_rec_x = [e["rec_x_mean"] for e in iter_experiments]
    if args.x_mode == "shared_epochs":
        iter_org_x = [e["split_iter"] / args.iters_per_epoch for e in iter_experiments]
        iter_rec_x = [e["split_iter"] / args.iters_per_epoch for e in iter_experiments]

    ax.scatter(
        iter_org_x,
        [e["org_rel_barrier"] for e in iter_experiments],
        c=iter_org_color,
        marker="s",
        s=90,
        facecolors="none",
        linewidths=1.8,
        label="Original (iter no-aug)",
        zorder=6,
    )

    if args.iter_overlay_mode == "all":
        first_seed_label = True
        for e in iter_experiments:
            label = "Recovered WM seeds (iter no-aug)" if first_seed_label else None
            first_seed_label = False
            x_seed = e["rec_xs"] if args.x_mode == "distance" else [e["split_iter"] / args.iters_per_epoch] * len(e["rec_xs"])
            ax.scatter(
                x_seed,
                e["rec_ys"],
                c=iter_rec_color,
                marker="s",
                s=28,
                alpha=0.30,
                linewidths=0,
                label=label,
                zorder=4,
            )

        ax.errorbar(
            iter_rec_x,
            [e["rec_y_mean"] for e in iter_experiments],
            xerr=None if args.x_mode == "shared_epochs" else [e["rec_x_std"] for e in iter_experiments],
            yerr=[e["rec_y_std"] for e in iter_experiments],
            fmt="s",
            color=iter_rec_color,
            ecolor=iter_rec_color,
            elinewidth=2.0,
            capsize=5,
            capthick=1.8,
            markersize=8,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label="Recovered mean ± std (iter no-aug)",
            zorder=7,
        )

    for e, x0, x1 in zip(iter_experiments, iter_org_x, iter_rec_x):
        ax.annotate(
            e["epoch_label"],
            (x0, e["org_rel_barrier"]),
            textcoords="offset points",
            xytext=(3, -12),
            fontsize=8,
            color=iter_org_color,
        )
        if args.iter_overlay_mode == "all":
            ax.annotate(
                "",
                xy=(x1, e["rec_y_mean"]),
                xytext=(x0, e["org_rel_barrier"]),
                arrowprops=dict(arrowstyle="->", color=iter_rec_color, alpha=0.75, lw=1.5),
            )

    if args.x_mode == "distance":
        ax.set_xlabel("L2 Distance", fontsize=12)
    else:
        ax.set_xlabel("Shared Training Epochs", fontsize=12)
    ax.set_ylabel("Relative Train-Accuracy Barrier", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center")

    plt.tight_layout()
    output_path = PROJECT_ROOT / args.legacy_output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved combined plot to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
