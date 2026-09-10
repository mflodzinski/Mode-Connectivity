"""Create a 2D pseudo loss landscape with an explicit interpolation barrier."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "plots" / "pseudo_loss_landscape_two_minima" / "pseudo_loss_landscape_two_minima_barrier_paths.png"


def loss_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Synthetic two-minima surface with a pronounced central loss barrier."""
    bowl = 0.055 * (x**2 + 0.90 * y**2) + 0.018 * (x**2 + y**2) ** 2
    well_a = 1.40 * np.exp(-(((x + 1.25) / 0.52) ** 2 + ((y + 0.95) / 0.48) ** 2))
    well_b = 1.38 * np.exp(-(((x - 1.25) / 0.52) ** 2 + ((y - 0.85) / 0.48) ** 2))

    # The ridge is intentionally high so the direct interpolation crosses
    # a yellow region comparable to random initialization loss.
    central_ridge = 2.10 * np.exp(-(((x - 0.02) / 0.62) ** 2 + ((y + 0.03) / 0.52) ** 2))
    ridge_tail = 0.55 * np.exp(-(((x + 0.15) / 1.10) ** 2 + ((y - 0.05) / 0.95) ** 2))
    return bowl - well_a - well_b + central_ridge + ridge_tail + 1.10


def sample_polyline(points: list[tuple[float, float]], samples_per_segment: int = 60) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        xs.append((1.0 - t) * start[0] + t * end[0])
        ys.append((1.0 - t) * start[1] + t * end[1])
    xs.append(np.array([points[-1][0]]))
    ys.append(np.array([points[-1][1]]))
    return np.concatenate(xs), np.concatenate(ys)


def add_path_arrows(ax, x: np.ndarray, y: np.ndarray, color: str, positions: tuple[float, ...], size: int = 22) -> None:
    for pos in positions:
        idx = int(pos * (len(x) - 2))
        ax.annotate(
            "",
            xy=(x[idx + 1], y[idx + 1]),
            xytext=(x[idx - 7], y[idx - 7]),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=0, mutation_scale=size),
            zorder=12,
        )


def label(ax, x: float, y: float, text: str, ha: str = "center", va: str = "center", size: int = 20) -> None:
    txt = ax.text(x, y, text, color="white", fontsize=size, weight="bold", ha=ha, va=va, zorder=20)
    txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground="black", alpha=0.45)])


def main() -> None:
    x = np.linspace(-2.55, 2.55, 420)
    y = np.linspace(-2.35, 2.35, 420)
    xx, yy = np.meshgrid(x, y)
    zz = loss_surface(xx, yy)

    minimum_a = (-1.25, -0.95)
    minimum_b = (1.25, 0.85)
    init_a = (-2.05, 0.90)
    init_b = (2.00, 1.85)

    opt_a_x, opt_a_y = sample_polyline([init_a, (-1.75, 0.65), (-1.45, 0.30), (-1.18, -0.18), minimum_a], 65)
    opt_b_x, opt_b_y = sample_polyline([init_b, (1.78, 1.45), (1.52, 1.15), minimum_b], 65)

    t = np.linspace(0.0, 1.0, 180)
    lin_x = (1.0 - t) * minimum_a[0] + t * minimum_b[0]
    lin_y = (1.0 - t) * minimum_a[1] + t * minimum_b[1]

    nonlinear_x, nonlinear_y = sample_polyline(
        [minimum_a, (-1.05, -1.52), (-0.25, -1.72), (0.65, -1.38), (1.10, -0.20), minimum_b],
        75,
    )

    fig, ax = plt.subplots(figsize=(13.2, 9.4), dpi=180)

    levels = np.linspace(np.percentile(zz, 1), np.percentile(zz, 99.4), 38)
    filled = ax.contourf(xx, yy, zz, levels=levels, cmap="viridis", extend="both")
    ax.contour(xx, yy, zz, levels=levels, colors="black", linewidths=0.55, alpha=0.33)

    ax.plot(opt_a_x, opt_a_y, color="#ffd33d", lw=4.0, solid_capstyle="round", zorder=10)
    ax.plot(opt_b_x, opt_b_y, color="#ff8c00", lw=4.0, solid_capstyle="round", zorder=10)
    add_path_arrows(ax, opt_a_x, opt_a_y, "#ffd33d", (0.55, 0.78))
    add_path_arrows(ax, opt_b_x, opt_b_y, "#ff8c00", (0.62,))

    ax.plot(lin_x, lin_y, color="#e63946", lw=4.4, solid_capstyle="round", zorder=14)
    add_path_arrows(ax, lin_x, lin_y, "#e63946", (0.52,), size=24)

    ax.plot(nonlinear_x, nonlinear_y, color="#35c759", lw=4.4, solid_capstyle="round", zorder=14)
    add_path_arrows(ax, nonlinear_x, nonlinear_y, "#35c759", (0.40, 0.72), size=24)

    ax.scatter(*init_a, marker="*", s=420, color="#ffd33d", edgecolor="black", linewidth=1.6, zorder=15)
    ax.scatter(*init_b, marker="*", s=420, color="#ff8c00", edgecolor="black", linewidth=1.6, zorder=15)
    ax.scatter(*minimum_a, marker="X", s=300, color="#ffd33d", edgecolor="black", linewidth=1.6, zorder=15)
    ax.scatter(*minimum_b, marker="X", s=300, color="#ff8c00", edgecolor="black", linewidth=1.6, zorder=15)

    label(ax, init_a[0] - 0.20, init_a[1] + 0.18, "Init A", ha="left")
    label(ax, init_b[0] - 0.12, init_b[1] + 0.18, "Init B", ha="center")
    label(ax, minimum_a[0] - 0.10, minimum_a[1] - 0.25, "Minimum A", ha="left")
    label(ax, minimum_b[0] - 0.28, minimum_b[1] - 0.25, "Minimum B", ha="left")
    label(ax, 0.05, 0.22, "high-loss\nbarrier", size=18)

    legend_handles = [
        Line2D([0], [0], color="#e63946", lw=4.4, label="linear interpolation"),
        Line2D([0], [0], color="#35c759", lw=4.4, label="nonlinear low-loss path"),
        Line2D([0], [0], color="#ffd33d", lw=4.0, label="optimization paths"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, framealpha=0.92, fontsize=14)

    cbar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.024)
    cbar.set_ticks([levels[0], levels[-1]])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlim(-2.50, 2.50)
    ax.set_ylim(-2.30, 2.30)
    ax.set_xlabel("$u_1$", fontsize=26)
    ax.set_ylabel("$u_2$", fontsize=26)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
