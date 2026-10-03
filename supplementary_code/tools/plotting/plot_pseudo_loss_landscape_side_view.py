"""Create a side-view companion for the pseudo loss landscape figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import art3d


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "plots" / "pseudo_loss_landscape_two_minima" / "pseudo_loss_landscape_two_minima_side_view.png"


def loss_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Synthetic two-basin loss surface used for the explanatory figure."""
    bowl = 0.030 * (x**2 + y**2) ** 2 + 0.070 * (x**2 + 0.70 * y**2)
    well_a = 1.15 * np.exp(-(((x + 1.35) / 0.55) ** 2 + ((y + 1.10) / 0.50) ** 2))
    well_b = 1.12 * np.exp(-(((x - 1.25) / 0.55) ** 2 + ((y - 0.95) / 0.50) ** 2))
    ridge = 0.45 * np.exp(-(((x - 0.10) / 0.65) ** 2 + ((y + 0.05) / 0.75) ** 2))
    tilt = 0.025 * x - 0.015 * y
    return bowl - well_a - well_b + ridge + tilt + 1.35


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


def annotate_point(
    ax,
    x: float,
    y: float,
    text: str,
    color: str,
    marker: str,
    dz: float = 0.16,
    dx: float = 0.0,
    dy: float = 0.0,
) -> None:
    z = float(loss_surface(np.array(x), np.array(y)))
    ax.scatter([x], [y], [z + 0.03], marker=marker, s=190, color=color, edgecolor="black", linewidth=1.2, zorder=20)
    label = ax.text(x + dx, y + dy, z + dz, text, color="white", fontsize=13, weight="bold", ha="center", va="bottom", zorder=30)
    label.set_path_effects([pe.withStroke(linewidth=2.8, foreground="black", alpha=0.55)])


def main() -> None:
    x = np.linspace(-2.45, 2.45, 180)
    y = np.linspace(-2.20, 2.20, 180)
    xx, yy = np.meshgrid(x, y)
    zz = loss_surface(xx, yy)

    minimum_a = (-1.35, -1.10)
    minimum_b = (1.25, 0.95)
    init_a = (-2.05, 0.78)
    init_b = (2.02, 1.72)

    opt_a_x, opt_a_y = sample_polyline([init_a, (-1.70, 0.45), (-1.35, 0.10), (-1.28, -0.38), minimum_a])
    opt_b_x, opt_b_y = sample_polyline([init_b, (1.78, 1.25), (1.45, 0.88), minimum_b])

    t = np.linspace(0.0, 1.0, 160)
    lin_x = (1.0 - t) * minimum_a[0] + t * minimum_b[0]
    lin_y = (1.0 - t) * minimum_a[1] + t * minimum_b[1]
    lin_z = loss_surface(lin_x, lin_y)

    curve_x, curve_y = sample_polyline([minimum_a, (-0.75, -1.35), (0.20, -1.48), (0.95, -0.15), minimum_b], 70)
    curve_z = loss_surface(curve_x, curve_y)

    fig = plt.figure(figsize=(14.2, 8.0), dpi=180)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92, rcount=130, ccount=130)
    ax.contour(xx, yy, zz, zdir="z", offset=zz.min() - 0.08, levels=24, cmap="viridis", linewidths=0.75, alpha=0.55)

    ax.plot(opt_a_x, opt_a_y, loss_surface(opt_a_x, opt_a_y) + 0.045, color="#ffd33d", linewidth=3.2, zorder=12)
    ax.plot(opt_b_x, opt_b_y, loss_surface(opt_b_x, opt_b_y) + 0.045, color="#ff8c00", linewidth=3.2, zorder=12)

    ax.plot(lin_x, lin_y, lin_z + 0.13, color="#e63946", linewidth=4.8, zorder=18)
    ax.plot(curve_x, curve_y, curve_z + 0.10, color="#34c759", linewidth=3.8, zorder=15)

    midpoint_idx = int(np.argmax(lin_z))
    mid_x = float(lin_x[midpoint_idx])
    mid_y = float(lin_y[midpoint_idx])
    mid_z = float(lin_z[midpoint_idx])
    endpoint_baseline = max(float(loss_surface(np.array(minimum_a[0]), np.array(minimum_a[1]))), float(loss_surface(np.array(minimum_b[0]), np.array(minimum_b[1]))))
    ax.plot([mid_x, mid_x], [mid_y, mid_y], [endpoint_baseline, mid_z + 0.13], color="#e63946", linewidth=2.6, linestyle="--")
    barrier_label = ax.text(mid_x - 0.28, mid_y - 0.18, mid_z + 0.42, "linear barrier", color="#e63946", fontsize=13, weight="bold")
    barrier_label.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white", alpha=0.85)])

    annotate_point(ax, *init_a, "Init A", "#ffd33d", "*", dz=0.16, dx=-0.08, dy=0.00)
    annotate_point(ax, *init_b, "Init B", "#ff8c00", "*", dz=0.16, dx=-0.18, dy=-0.10)
    annotate_point(ax, *minimum_a, "Minimum A", "#ffd33d", "X", dz=0.14, dx=-0.15, dy=-0.06)
    annotate_point(ax, *minimum_b, "Minimum B", "#ff8c00", "X", dz=0.14, dx=0.00, dy=-0.13)

    # A translucent vertical curtain under the interpolation makes the side-view barrier legible.
    verts = []
    floor = zz.min() - 0.08
    for i in range(len(lin_x) - 1):
        verts.append(
            [
                (lin_x[i], lin_y[i], floor),
                (lin_x[i + 1], lin_y[i + 1], floor),
                (lin_x[i + 1], lin_y[i + 1], lin_z[i + 1] + 0.04),
                (lin_x[i], lin_y[i], lin_z[i] + 0.04),
            ]
        )
    curtain = art3d.Poly3DCollection(verts, facecolor="#e63946", alpha=0.13, edgecolor="none")
    ax.add_collection3d(curtain)

    ax.view_init(elev=21, azim=-58)
    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-2.20, 2.20)
    ax.set_zlim(floor, zz.max() + 0.25)
    ax.set_xlabel("$u_1$", fontsize=18, labelpad=8)
    ax.set_ylabel("$u_2$", fontsize=18, labelpad=8)
    ax.set_zlabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.xaxis.pane.set_alpha(0.0)
    ax.yaxis.pane.set_alpha(0.0)
    ax.zaxis.pane.set_alpha(0.0)

    legend_handles = [
        Line2D([0], [0], color="#e63946", lw=4.8, label="direct linear interpolation"),
        Line2D([0], [0], color="#34c759", lw=3.8, label="curved low-loss path"),
        Line2D([0], [0], color="#e63946", lw=2.6, linestyle="--", label="barrier height"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.04, 0.92), frameon=False, fontsize=12)

    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.02, top=0.98)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
