"""Plot test-error barriers for the pytorch-vgg WM split suite."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

PAIR_SPECS = [
    ("100/100", 100.0, "results/analysis/pytorch_vgg_split_wm/100_100/results.json"),
    ("80/120", 80.0, "results/analysis/pytorch_vgg_split_wm/80_120/results.json"),
    ("30/170", 30.0, "results/analysis/pytorch_vgg_split_wm/30_170/results.json"),
    ("8/192", 8.0, "results/analysis/pytorch_vgg_split_wm/8_192/results.json"),
    ("4/196", 4.0, "results/analysis/pytorch_vgg_split_wm/4_196/results.json"),
    ("3/197", 3.0, "results/analysis/pytorch_vgg_split_wm/3_197/results.json"),
    ("0/200", 0.0, "results/analysis/pytorch_vgg_split_wm/0_200/results.json"),
    ("independent", None, "results/analysis/pytorch_vgg_split_wm/independent/results.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot test-error barrier vs distance for the pytorch-vgg WM suite.")
    parser.add_argument(
        "--output",
        type=str,
        default="plots/pytorch_vgg_split_suite_wm_test_acc_relative_barrier_vs_distance.png",
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="plots/pytorch_vgg_split_suite_wm_test_acc_relative_barrier_vs_distance.csv",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r") as handle:
        return json.load(handle)


def acc_barrier(values: list[float]) -> float:
    endpoint_avg = 0.5 * (values[0] + values[-1])
    return endpoint_avg - min(values)


def load_rows() -> list[dict]:
    rows = []
    for label, shared_epochs, json_rel in PAIR_SPECS:
        json_path = PROJECT_ROOT / json_rel
        if not json_path.exists():
            continue
        payload = load_json(json_path)
        rows.append(
            {
                "label": label,
                "shared_epochs": shared_epochs,
                "original_l2_distance": float(payload["distances"]["w0_w1"]["l2_distance"]),
                "recovered_l2_distance": float(payload["distances"]["w0_w1_recovered"]["l2_distance"]),
                "original_test_error_barrier": acc_barrier(payload["original_barrier"]["test_acc"]),
                "recovered_test_error_barrier": acc_barrier(payload["recovered_barrier_to_w0"]["test_acc"]),
            }
        )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "shared_epochs",
        "original_l2_distance",
        "recovered_l2_distance",
        "original_test_error_barrier",
        "recovered_test_error_barrier",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict], output_path: Path) -> None:
    shared = [row for row in rows if row["shared_epochs"] is not None]
    independent = [row for row in rows if row["shared_epochs"] is None]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    orig_color = "#1f77b4"
    rec_color = "#2ca02c"

    ax.scatter(
        [row["original_l2_distance"] for row in shared],
        [row["original_test_error_barrier"] for row in shared],
        color=orig_color,
        s=80,
        label="Shared init",
        zorder=5,
    )
    ax.scatter(
        [row["recovered_l2_distance"] for row in shared],
        [row["recovered_test_error_barrier"] for row in shared],
        color=rec_color,
        s=80,
        marker="s",
        label="Shared init + perm. + WM",
        zorder=6,
    )

    for row in shared:
        if row["label"] == "0/200":
            xytext = (-10, -2)
            ha = "right"
        elif row["label"] == "8/192":
            xytext = (1, 5)
            ha = "left"
        else:
            xytext = (4, 5)
            ha = "left"
        ax.annotate(
            row["label"],
            (row["original_l2_distance"], row["original_test_error_barrier"]),
            textcoords="offset points",
            xytext=xytext,
            fontsize=8,
            fontweight="bold",
            ha=ha,
        )
        ax.annotate(
            "",
            xy=(row["recovered_l2_distance"], row["recovered_test_error_barrier"]),
            xytext=(row["original_l2_distance"], row["original_test_error_barrier"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#666666",
                alpha=0.85,
                lw=1.3,
                linestyle=(0, (2.0, 2.0)),
                mutation_scale=13,
            ),
        )

    independent_orig_label = True
    independent_wm_label = True
    for row in independent:
        ax.scatter(
            [row["original_l2_distance"]],
            [row["original_test_error_barrier"]],
            color="#d62728",
            marker="o",
            s=110,
            edgecolors="black",
            linewidths=1.0,
            zorder=7,
            label="Indep. init" if independent_orig_label else None,
        )
        independent_orig_label = False
        ax.scatter(
            [row["recovered_l2_distance"]],
            [row["recovered_test_error_barrier"]],
            color="#d62728",
            marker="s",
            s=110,
            edgecolors="black",
            linewidths=1.0,
            zorder=8,
            label="Indep. init + WM" if independent_wm_label else None,
        )
        independent_wm_label = False
        ax.annotate(
            "",
            xy=(row["recovered_l2_distance"], row["recovered_test_error_barrier"]),
            xytext=(row["original_l2_distance"], row["original_test_error_barrier"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color="#666666",
                alpha=0.85,
                lw=1.3,
                linestyle=(0, (2.0, 2.0)),
                mutation_scale=13,
            ),
        )

    ax.set_xlabel("L2 Distance", fontsize=12)
    ax.set_ylabel("Test Error Barrier", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_rows()
    if not rows:
        raise RuntimeError("No WM benchmark result files found for pytorch-vgg split suite.")
    write_csv(rows, PROJECT_ROOT / args.csv_output)
    plot(rows, PROJECT_ROOT / args.output)


if __name__ == "__main__":
    main()
