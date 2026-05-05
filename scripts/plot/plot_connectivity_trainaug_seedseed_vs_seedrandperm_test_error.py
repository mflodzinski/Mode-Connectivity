"""Plot Test Error panel for trainaug seed-seed vs seed-randperm curves."""

from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def load_curve(npz_path: str) -> dict:
    data = np.load(npz_path)
    return {
        "ts": data["ts"],
        "te_err": data["te_err"],
        "te_loss": data["te_loss"],
        "tr_err": data["tr_err"],
        "tr_loss": data["tr_loss"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot trainaug seed-seed vs seed-randperm Test Error.")
    parser.add_argument(
        "--output",
        type=str,
        default="plots/connectivity_trainaug_seedseed_vs_seedrandperm_test_error.png",
        help="Output file path.",
    )
    args = parser.parse_args()

    seed_paths = [
        "results/vgg16/cifar10/curves/standard_trainaug/seed0-seed1_reg/evaluations/curve.npz",
        "results/vgg16/cifar10/curves/standard_trainaug/seed0-seed2_bezier/evaluations/curve.npz",
        "results/vgg16/cifar10/curves/standard_trainaug/seed1-seed2_bezier/evaluations/curve.npz",
    ]
    randperm_paths = [
        "results/vgg16/cifar10/curves/standard_trainaug/seed0-randperm_reg/evaluations/curve.npz",
        "results/vgg16/cifar10/curves/standard_trainaug/seed1-randperm_reg/evaluations/curve.npz",
        "results/vgg16/cifar10/curves/standard_trainaug/seed2-randperm_reg/evaluations/curve.npz",
    ]

    seed_curves = [load_curve(os.path.join(PROJECT_ROOT, path)) for path in seed_paths]
    randperm_curves = [load_curve(os.path.join(PROJECT_ROOT, path)) for path in randperm_paths]

    fig, ax = plt.subplots(figsize=(7.8, 6.0))

    seed_color = "#1f77b4"
    randperm_color = "#ff7f0e"
    short_dash = (0, (2.2, 2.2))

    for idx, curve in enumerate(seed_curves):
        ax.plot(
            curve["ts"],
            curve["te_err"],
            color=seed_color,
            linewidth=2,
            linestyle=short_dash,
            dash_capstyle="butt",
            label="Different modes (curve)" if idx == 0 else None,
        )

    for idx, curve in enumerate(randperm_curves):
        ax.plot(
            curve["ts"],
            curve["te_err"],
            color=randperm_color,
            linewidth=2,
            linestyle=":",
            label="Permuted modes (curve)" if idx == 0 else None,
        )

    ax.set_xlabel("t (interpolation parameter)", fontsize=11)
    ax.set_ylabel("Test Error (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)
    ax.legend(fontsize=9, loc="upper right", frameon=True)

    fig.tight_layout()

    output_path = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
