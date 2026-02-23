#!/usr/bin/env python3
"""Create a 2D pseudo loss landscape with multiple minima and optimization paths."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pseudo_loss(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Synthetic loss surface with three basins and a central barrier."""
    poly = 0.015 * (x**4 + y**4) + 0.04 * (x**2 + y**2)

    basin_1 = -2.3 * np.exp(-((x + 1.9) ** 2 + (y + 1.1) ** 2) / 1.3)
    basin_2 = -2.0 * np.exp(-((x - 2.1) ** 2 + (y - 1.6) ** 2) / 1.1)
    basin_3 = -2.1 * np.exp(-((x - 3.2) ** 2 + (y - 3.0) ** 2) / 0.9)
    barrier = 1.1 * np.exp(-((x - 0.1) ** 2 + (y - 0.2) ** 2) / 1.8)

    tilt = 0.015 * x - 0.01 * y
    return poly + basin_1 + basin_2 + basin_3 + barrier + tilt


def finite_difference_grad(x: float, y: float, eps: float = 1e-3) -> np.ndarray:
    dfdx = (pseudo_loss(x + eps, y) - pseudo_loss(x - eps, y)) / (2.0 * eps)
    dfdy = (pseudo_loss(x, y + eps) - pseudo_loss(x, y - eps)) / (2.0 * eps)
    return np.array([dfdx, dfdy], dtype=float)


def gradient_descent_path(
    start: tuple[float, float],
    lr: float = 0.08,
    steps: int = 120,
) -> np.ndarray:
    point = np.array(start, dtype=float)
    path = [point.copy()]
    for _ in range(steps):
        grad = finite_difference_grad(point[0], point[1])
        point = point - lr * grad
        path.append(point.copy())
    return np.array(path)


def draw(output_path: Path) -> None:
    x = np.linspace(-4.5, 4.5, 420)
    y = np.linspace(-4.5, 4.5, 420)
    xx, yy = np.meshgrid(x, y)
    zz = pseudo_loss(xx, yy)
    zz_min = float(np.min(zz))
    zz_range = float(np.ptp(zz))
    if zz_range == 0.0:
        zz_display = np.zeros_like(zz)
    else:
        zz_display = 5.0 * (zz - zz_min) / zz_range

    start_a = (-3.6, 2.4)
    start_c = (3.5, 3.9)

    path_a = gradient_descent_path(start_a, lr=0.078, steps=130)
    path_c = gradient_descent_path(start_c, lr=0.078, steps=130)

    min_a = path_a[-1]
    min_c = path_c[-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    filled = ax.contourf(xx, yy, zz_display, levels=np.linspace(0.0, 5.0, 45), cmap="viridis")
    ax.contour(
        xx,
        yy,
        zz_display,
        levels=np.linspace(0.0, 5.0, 25),
        colors="black",
        alpha=0.35,
        linewidths=0.6,
    )
    cbar = fig.colorbar(filled, ax=ax, pad=0.02)
    cbar.set_ticks(np.linspace(0.0, 5.0, 6))

    ax.plot(path_a[:, 0], path_a[:, 1], color="#f4c430", lw=2.5)
    ax.plot(path_c[:, 0], path_c[:, 1], color="#ff8c00", lw=2.5)

    # Place directional arrows along each trajectory.
    arrow_idx = [16, 34, 58, 86, 112]
    for i in arrow_idx:
        if i + 1 < len(path_a):
            ax.annotate(
                "",
                xy=path_a[i + 1],
                xytext=path_a[i],
                arrowprops=dict(arrowstyle="->", color="#f4c430", lw=1.8),
            )
        if i + 1 < len(path_c):
            ax.annotate(
                "",
                xy=path_c[i + 1],
                xytext=path_c[i],
                arrowprops=dict(arrowstyle="->", color="#ff8c00", lw=1.8),
            )

    ax.scatter(*start_a, marker="*", s=260, color="#f4c430", edgecolor="black", zorder=6)
    ax.scatter(*start_c, marker="*", s=260, color="#ff8c00", edgecolor="black", zorder=6)
    ax.scatter(*min_a, marker="X", s=180, color="#f4c430", edgecolor="black", zorder=7)
    ax.scatter(*min_c, marker="X", s=180, color="#ff8c00", edgecolor="black", zorder=7)

    ax.text(
        start_a[0] - 0.35, start_a[1] + 0.25, "Init A", fontsize=11, weight="bold", color="white"
    )
    ax.text(
        start_c[0] - 0.2, start_c[1] + 0.25, "Init B", fontsize=11, weight="bold", color="white"
    )
    ax.text(
        min_a[0] - 0.2, min_a[1] - 0.45, "Minimum A", fontsize=10, weight="bold", color="white"
    )
    ax.text(
        min_c[0] - 0.5, min_c[1] - 0.45, "Minimum B", fontsize=10, weight="bold", color="white"
    )

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-4.5, 4.5)
    ax.grid(alpha=0.15)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plots/pseudo_loss_landscape_two_minima.png"),
        help="Output image path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw(args.output)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
