"""Plot Test Error panel with mixed source roots.

Main lines:
- different-modes curve and inset seed-seed curves from standard_trainaug
- mirrored curve/linear and inset seed-mirror curves from standard

The different-modes linear path falls back to the standard source because
``standard_trainaug/seed0-seed1_reg/evaluations`` does not contain ``linear.npz``.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)


def load_npz(npz_path: str) -> dict:
    data = np.load(npz_path)
    return {
        "ts": data["ts"],
        "tr_loss": data["tr_loss"],
        "te_loss": data["te_loss"],
        "tr_err": data["tr_err"],
        "te_err": data["te_err"],
    }


def add_inset(ax, seed_curves: dict, mirror_curves: dict) -> None:
    inset_ax = ax.inset_axes([0.38, 0.14, 0.57, 0.74])
    for curve_data in seed_curves.values():
        inset_ax.plot(
            curve_data["ts"],
            curve_data["te_err"],
            color="#1f77b4",
            linewidth=2,
            linestyle=(0, (2.2, 2.2)),
            dash_capstyle="butt",
        )
    for curve_data in mirror_curves.values():
        inset_ax.plot(curve_data["ts"], curve_data["te_err"], ":", color="#ff7f0e", linewidth=2)

    seed_peak = max(float(np.max(curve_data["te_err"])) for curve_data in seed_curves.values())
    mirror_peak = max(float(np.max(curve_data["te_err"])) for curve_data in mirror_curves.values())
    endpoint_ref = float(next(iter(seed_curves.values()))["te_err"][0])
    y_min = min(float(np.min(curve_data["te_err"])) for curve_data in list(seed_curves.values()) + list(mirror_curves.values()))
    y_max = max(seed_peak, mirror_peak)
    y_pad = max((y_max - y_min) * 0.08, 1e-4)

    inset_ax.set_xlim(0.0, 1.0)
    inset_ax.set_ylim(y_min - y_pad, y_max + y_pad)
    inset_ax.set_xticks([])
    inset_ticks = sorted({endpoint_ref, seed_peak, mirror_peak})
    inset_ax.set_yticks(inset_ticks)
    inset_ax.set_yticklabels(
        ["7.0" if abs(value - endpoint_ref) < 1e-9 else f"{value:.1f}" for value in inset_ticks],
        fontsize=8,
    )
    inset_ax.grid(True, alpha=0.18)
    for spine in inset_ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_alpha(0.8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mixed-source Test Error panel.")
    parser.add_argument(
        "--output",
        type=str,
        default="plots/connectivity_reg_comparison_with_linear_yaxis_test_error_mixed_sources.png",
        help="Output file path.",
    )
    args = parser.parse_args()

    paths = {
        "diff_curve": "results/vgg16/cifar10/curves/standard_trainaug/seed0-seed1_reg/evaluations/curve.npz",
        "diff_linear": "results/vgg16/cifar10/curves/standard/seed0-seed1_reg/evaluations/linear.npz",
        "perm0_curve": "results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/curve.npz",
        "perm0_linear": "results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/linear.npz",
        "perm1_curve": "results/vgg16/cifar10/curves/standard/seed1-mirror_reg/evaluations/curve.npz",
        "perm1_linear": "results/vgg16/cifar10/endpoints/standard/seed1_mirrored/evaluations/linear.npz",
    }

    seed_curve_paths = {
        "seed0-seed1": "results/vgg16/cifar10/curves/standard_trainaug/seed0-seed1_reg/evaluations/curve.npz",
        "seed0-seed2": "results/vgg16/cifar10/curves/standard_trainaug/seed0-seed2_bezier/evaluations/curve.npz",
        "seed1-seed2": "results/vgg16/cifar10/curves/standard_trainaug/seed1-seed2_bezier/evaluations/curve.npz",
    }
    mirror_curve_paths = {
        "seed0-mirror": "results/vgg16/cifar10/curves/standard/seed0-mirror_reg/evaluations/curve.npz",
        "seed1-mirror": "results/vgg16/cifar10/curves/standard/seed1-mirror_reg/evaluations/curve.npz",
    }

    data = {}
    for key, rel_path in paths.items():
        full_path = os.path.join(PROJECT_ROOT, rel_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(full_path)
        data[key] = load_npz(full_path)

    seed_curves = {name: load_npz(os.path.join(PROJECT_ROOT, rel_path)) for name, rel_path in seed_curve_paths.items()}
    mirror_curves = {name: load_npz(os.path.join(PROJECT_ROOT, rel_path)) for name, rel_path in mirror_curve_paths.items()}

    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    diff_color = "#1f77b4"
    mirror_color = "#ff7f0e"

    ax.plot(
        data["diff_curve"]["ts"],
        data["diff_curve"]["te_err"],
        color=diff_color,
        linewidth=2,
        linestyle=(0, (2.2, 2.2)),
        dash_capstyle="butt",
        label="Different modes (curve)",
    )
    ax.plot(data["diff_linear"]["ts"], data["diff_linear"]["te_err"], "-", color=diff_color, linewidth=2, label="Different modes (linear)")
    ax.plot(data["perm0_curve"]["ts"], data["perm0_curve"]["te_err"], ":", color=mirror_color, linewidth=2, label="Mirrored modes (curve)")
    ax.plot(data["perm0_linear"]["ts"], data["perm0_linear"]["te_err"], "-", color=mirror_color, linewidth=2, label="Mirrored modes (linear)")
    ax.plot(data["perm1_curve"]["ts"], data["perm1_curve"]["te_err"], ":", color=mirror_color, linewidth=2)
    ax.plot(data["perm1_linear"]["ts"], data["perm1_linear"]["te_err"], "-", color=mirror_color, linewidth=2)

    ax.set_xlabel("t (interpolation parameter)", fontsize=11)
    ax.set_ylabel("Test Error (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, alpha=0.3)

    add_inset(ax, seed_curves, mirror_curves)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        fontsize=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=True,
    )

    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.20, top=0.97)

    output_path = os.path.join(PROJECT_ROOT, args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
