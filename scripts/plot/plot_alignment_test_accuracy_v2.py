"""
Plot relative test-accuracy barrier: original vs recovered after alignment.

Compares the relative barrier

    (endpoint_avg - min(accs)) / (endpoint_avg + eps)

along the interpolation path before and after permutation alignment for
different split configurations.

V2: Includes starting point for the seed0-seed1 experiment.
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

EPS = 1e-12


def load_benchmark_data(json_path):
    """Load benchmark results from JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def relative_barrier(accs, eps=EPS):
    endpoint_avg = 0.5 * (accs[0] + accs[-1])
    return (endpoint_avg - min(accs)) / (endpoint_avg + eps)


def main():
    parser = argparse.ArgumentParser(description="Plot relative test-accuracy barrier: original vs recovered")
    parser.add_argument(
        "--output",
        type=str,
        default="plots/alignment_test_accuracy_v2.png",
        help="Output file path",
    )
    parser.add_argument("--show", action="store_true", help="Show plot interactively")
    args = parser.parse_args()

    data_files = {
        "150/50": "results/analysis/alignment_benchmark/results.json",
        "80/120": "results/analysis/alignment_benchmark_80split/results.json",
        "30/170": "results/analysis/alignment_benchmark_30split/results.json",
        "8/192": "results/analysis/alignment_benchmark_8split/results.json",
        "0/200": "results/analysis/alignment_benchmark_0split/results.json",
    }
    seed_file = "results/analysis/alignment_independent/seed0_seed1_results.json"

    experiments = []

    for name, path in data_files.items():
        full_path = os.path.join(project_root, path)
        if os.path.exists(full_path):
            data = load_benchmark_data(full_path)
            experiments.append(
                {
                    "name": name,
                    "w0_w1": data["distances"]["w0_w1"]["l2_distance"],
                    "w0_w1_recovered": data["distances"]["w0_w1_recovered"]["l2_distance"],
                    "org_rel_barrier": relative_barrier(data["original_barrier"]["test_acc"]),
                    "rec_rel_barrier": relative_barrier(data["recovered_barrier_to_w0"]["test_acc"]),
                    "is_independent": False,
                }
            )
        else:
            print(f"Warning: {full_path} not found, skipping {name}")

    seed_path = os.path.join(project_root, seed_file)
    if os.path.exists(seed_path):
        data = load_benchmark_data(seed_path)
        experiments.append(
            {
                "name": "seed0-seed1",
                "w0_w1": data["before_alignment"]["distance"]["l2_distance"],
                "w0_w1_recovered": data["after_alignment"]["distance"]["l2_distance"],
                "org_rel_barrier": relative_barrier(data["before_alignment"]["barrier"]["test_acc"]),
                "rec_rel_barrier": relative_barrier(data["after_alignment"]["barrier"]["test_acc"]),
                "is_independent": True,
                "show_original": True,
            }
        )
    else:
        print(f"Warning: {seed_path} not found, skipping seed0-seed1")

    if not experiments:
        print("No data files found!")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    org_color = "#1f77b4"
    rec_color = "#2ca02c"
    ind_diff_init_color = "#d62728"

    lmc_exps = [e for e in experiments if not e.get("is_independent", False)]
    ind_diff_init_exps = [e for e in experiments if e.get("is_independent", False)]

    if lmc_exps:
        x_org = [e["w0_w1"] for e in lmc_exps]
        y_org = [e["org_rel_barrier"] for e in lmc_exps]
        ax.scatter(x_org, y_org, c=org_color, marker="o", s=100, label="Original (shared training)", zorder=5)

        x_rec = [e["w0_w1_recovered"] for e in lmc_exps]
        y_rec = [e["rec_rel_barrier"] for e in lmc_exps]
        ax.scatter(x_rec, y_rec, c=rec_color, marker="o", s=100, label="Recovered (shared training)", zorder=5)

        for e in lmc_exps:
            ax.annotate(
                e["name"],
                (e["w0_w1"], e["org_rel_barrier"]),
                textcoords="offset points",
                xytext=(-5, 8),
                fontsize=8,
                color=org_color,
            )
            ax.annotate(
                "",
                xy=(e["w0_w1_recovered"], e["rec_rel_barrier"]),
                xytext=(e["w0_w1"], e["org_rel_barrier"]),
                arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7, lw=1.5),
            )

    first_diff_init = True
    first_diff_init_orig = True
    if ind_diff_init_exps:
        for e in ind_diff_init_exps:
            if e.get("show_original", False):
                label_orig = "Original (diff init)" if first_diff_init_orig else None
                first_diff_init_orig = False
                ax.scatter(
                    [e["w0_w1"]],
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
                    xy=(e["w0_w1_recovered"], e["rec_rel_barrier"]),
                    xytext=(e["w0_w1"], e["org_rel_barrier"]),
                    arrowprops=dict(arrowstyle="->", color=ind_diff_init_color, alpha=0.7, lw=2),
                )

            label_rec = "Recovered (diff init)" if first_diff_init else None
            first_diff_init = False
            ax.scatter(
                [e["w0_w1_recovered"]],
                [e["rec_rel_barrier"]],
                c=ind_diff_init_color,
                marker="D",
                s=150,
                edgecolors="black",
                linewidths=1.5,
                label=label_rec,
                zorder=6,
            )

    ax.set_xlabel("L2 Distance", fontsize=12)
    ax.set_ylabel("Relative Test-Accuracy Barrier", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()

    output_path = os.path.join(project_root, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    if args.show:
        plt.show()

    print("\n" + "=" * 88)
    print("RELATIVE TEST-ACCURACY BARRIER SUMMARY")
    print("=" * 88)
    print(
        f"{'Experiment':<15} {'w0_w1':>10} {'Org Rel':>12} {'w0_w1_rec':>10} {'Rec Rel':>12} {'Δ Rel':>12}"
    )
    print("-" * 88)
    for e in experiments:
        marker = " *" if e.get("is_independent", False) else ""
        delta = e["rec_rel_barrier"] - e["org_rel_barrier"]
        print(
            f"{e['name']:<15} {e['w0_w1']:>10.2f} {e['org_rel_barrier']:>12.6f} "
            f"{e['w0_w1_recovered']:>10.2f} {e['rec_rel_barrier']:>12.6f} {delta:>+11.6f}{marker}"
        )
    print("-" * 88)
    print("* = independent (different random init)")


if __name__ == "__main__":
    main()
