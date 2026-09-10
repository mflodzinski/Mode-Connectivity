"""Create two-minima pseudo loss landscape variants.

All variants are generated from the same synthetic loss surface. They differ
only in labels, markers, and paths.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "plots" / "pseudo_loss_landscape_two_minima"
OUT_FULL = OUT_DIR / "pseudo_loss_landscape_two_minima.png"
OUT_CLEAN = OUT_DIR / "pseudo_loss_landscape_two_minima_clean.png"
OUT_INITS = OUT_DIR / "pseudo_loss_landscape_two_minima_inits_only.png"
OUT_MINIMA_ONLY = OUT_DIR / "pseudo_loss_landscape_two_minima_minima_only.png"
OUT_LINEAR_ONLY = OUT_DIR / "pseudo_loss_landscape_two_minima_high_barrier_linear_only.png"
OUT_PERMUTATION = OUT_DIR / "pseudo_loss_landscape_two_minima_permutation_applied.png"
OUT_PERMUTATION_LINEAR = OUT_DIR / "pseudo_loss_landscape_two_minima_permutation_linear_connected.png"
OUT_THREE_MINIMA_STAGE0 = OUT_DIR / "pseudo_loss_landscape_three_minima_0_theta_a_theta_b.png"
OUT_THREE_MINIMA_STAGE1 = OUT_DIR / "pseudo_loss_landscape_three_minima_1_theta_a_theta_b_curved_path.png"
OUT_THREE_MINIMA_STAGE2 = OUT_DIR / "pseudo_loss_landscape_three_minima_2_add_p_theta_a.png"
OUT_THREE_MINIMA_STAGE3 = OUT_DIR / "pseudo_loss_landscape_three_minima_3_two_curved_paths.png"
OUT_THREE_MINIMA_CURVED = OUT_DIR / "pseudo_loss_landscape_three_minima_curved_paths.png"

MINIMUM_A = np.array([-1.25, -0.95])
MINIMUM_B = np.array([1.25, 0.85])
PERMUTED_B = np.array([-0.88, -0.68])
PERMUTED_A = np.array([-0.18, 1.16])
BASIN_A_CENTER = np.array([-1.02, -0.64])
BASIN_B_CENTER = np.array([1.05, 0.54])
BASIN_PERMUTED_A_CENTER = np.array([-0.46, 0.85])
INIT_A = np.array([-2.05, 0.90])
INIT_B = np.array([2.00, 1.85])

PATH_A = np.array(
    [
        INIT_A,
        [-1.84, 0.72],
        [-1.63, 0.46],
        [-1.48, 0.10],
        [-1.38, -0.32],
        MINIMUM_A,
    ]
)
PATH_B = np.array(
    [
        INIT_B,
        [1.86, 1.58],
        [1.68, 1.32],
        [1.48, 1.08],
        MINIMUM_B,
    ]
)


def loss_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Synthetic landscape with two minima separated by a high-loss ridge."""
    bowl = 0.055 * (x**2 + 0.90 * y**2) + 0.018 * (x**2 + y**2) ** 2
    well_a = 1.40 * np.exp(-(((x + 1.25) / 0.52) ** 2 + ((y + 0.95) / 0.48) ** 2))
    well_b = 1.38 * np.exp(-(((x - 1.25) / 0.52) ** 2 + ((y - 0.85) / 0.48) ** 2))
    central_ridge = 2.10 * np.exp(-(((x - 0.02) / 0.62) ** 2 + ((y + 0.03) / 0.52) ** 2))
    ridge_tail = 0.55 * np.exp(-(((x + 0.15) / 1.10) ** 2 + ((y - 0.05) / 0.95) ** 2))
    return bowl - well_a - well_b + central_ridge + ridge_tail + 1.10


def flat_circular_well(x: np.ndarray, y: np.ndarray, center: np.ndarray, radius: float, depth: float) -> np.ndarray:
    distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
    return depth * np.exp(-((distance_squared / radius**2) ** 2))


def cubic_bezier_path(
    start: np.ndarray,
    control_1: np.ndarray,
    control_2: np.ndarray,
    end: np.ndarray,
    samples: int = 220,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return (
        (1.0 - t) ** 3 * start
        + 3.0 * (1.0 - t) ** 2 * t * control_1
        + 3.0 * (1.0 - t) * t**2 * control_2
        + t**3 * end
    )


def curved_path_ab() -> np.ndarray:
    return cubic_bezier_path(
        MINIMUM_A,
        np.array([-1.20, -2.05]),
        np.array([1.75, -1.35]),
        MINIMUM_B,
    )


def curved_path_pa() -> np.ndarray:
    return cubic_bezier_path(
        MINIMUM_A,
        np.array([-2.38, -0.62]),
        np.array([-1.70, 1.62]),
        PERMUTED_A,
    )


def curved_channel(x: np.ndarray, y: np.ndarray, path: np.ndarray, width: float, depth: float) -> np.ndarray:
    min_distance_squared = np.full_like(x, np.inf, dtype=float)
    for point in path[::3]:
        distance_squared = (x - point[0]) ** 2 + (y - point[1]) ** 2
        min_distance_squared = np.minimum(min_distance_squared, distance_squared)
    return depth * np.exp(-(min_distance_squared / width**2))


def three_minima_loss_surface(
    x: np.ndarray,
    y: np.ndarray,
    *,
    include_permuted_a: bool = True,
    include_ab_channel: bool = True,
    include_permuted_a_channel: bool = True,
) -> np.ndarray:
    """Synthetic landscape with three minima and broad curved low-loss channels."""
    bowl = 0.050 * (x**2 + 0.90 * y**2) + 0.012 * (x**2 + y**2) ** 2
    well_a = flat_circular_well(x, y, BASIN_A_CENTER, radius=0.56, depth=1.65)
    well_b = flat_circular_well(x, y, BASIN_B_CENTER, radius=0.55, depth=1.58)

    central_ridge = 1.32 * np.exp(-(((x - 0.08) / 0.58) ** 2 + ((y + 0.05) / 0.54) ** 2))
    ridge_top_left = 0.64 * np.exp(-(((x + 1.10) / 0.42) ** 2 + ((y - 0.38) / 0.52) ** 2))
    ridge_top_right = 0.46 * np.exp(-(((x - 0.62) / 0.58) ** 2 + ((y - 0.64) / 0.50) ** 2))

    loss = (
        bowl
        - well_a
        - well_b
        + central_ridge
        + ridge_top_left
        + ridge_top_right
        + 1.24
    )
    if include_permuted_a:
        well_pa = flat_circular_well(x, y, BASIN_PERMUTED_A_CENTER, radius=0.54, depth=1.55)
        loss -= well_pa
    if include_ab_channel:
        loss -= curved_channel(x, y, curved_path_ab(), width=0.13, depth=0.34)
    if include_permuted_a_channel:
        loss -= curved_channel(x, y, curved_path_pa(), width=0.13, depth=0.30)
    return loss


def draw_base(ax):
    x = np.linspace(-2.55, 2.55, 520)
    y = np.linspace(-2.35, 2.35, 520)
    xx, yy = np.meshgrid(x, y)
    zz = loss_surface(xx, yy)

    levels = np.linspace(np.percentile(zz, 0.5), np.percentile(zz, 99.3), 46)
    filled = ax.contourf(xx, yy, zz, levels=levels, cmap="viridis", extend="both")
    ax.contour(xx, yy, zz, levels=levels, colors="black", linewidths=0.55, alpha=0.34)

    ax.set_xlim(-2.50, 2.50)
    ax.set_ylim(-2.30, 2.30)
    ax.set_xlabel("$u_1$", fontsize=31)
    ax.set_ylabel("$u_2$", fontsize=31)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    return filled


def draw_three_minima_base(
    ax,
    *,
    include_permuted_a: bool = True,
    include_ab_channel: bool = True,
    include_permuted_a_channel: bool = True,
):
    x = np.linspace(-2.55, 2.55, 520)
    y = np.linspace(-2.35, 2.35, 520)
    xx, yy = np.meshgrid(x, y)
    zz = three_minima_loss_surface(
        xx,
        yy,
        include_permuted_a=include_permuted_a,
        include_ab_channel=include_ab_channel,
        include_permuted_a_channel=include_permuted_a_channel,
    )
    final_zz = three_minima_loss_surface(
        xx,
        yy,
        include_permuted_a=True,
        include_ab_channel=True,
        include_permuted_a_channel=True,
    )

    levels = np.linspace(np.percentile(final_zz, 0.5), np.percentile(final_zz, 99.3), 46)
    filled = ax.contourf(xx, yy, zz, levels=levels, cmap="viridis", extend="both")
    ax.contour(xx, yy, zz, levels=levels, colors="black", linewidths=0.55, alpha=0.34)

    ax.set_xlim(-2.50, 2.50)
    ax.set_ylim(-2.30, 2.30)
    ax.set_xlabel("$u_1$", fontsize=31)
    ax.set_ylabel("$u_2$", fontsize=31)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    return filled


def add_colorbar(fig, ax, filled) -> None:
    cbar = fig.colorbar(filled, ax=ax, fraction=0.046, pad=0.024)
    cbar.set_ticks([filled.levels[0], filled.levels[-1]])
    cbar.set_ticklabels(["Low", "High"])
    cbar.ax.tick_params(labelsize=17)


def add_init_markers(ax) -> None:
    ax.scatter(*INIT_A, marker="*", s=520, color="#ffd33d", edgecolor="black", linewidth=1.7, zorder=10)
    ax.scatter(*INIT_B, marker="*", s=520, color="#ff8c00", edgecolor="black", linewidth=1.7, zorder=10)

    label_style = dict(color="white", fontsize=28, weight="bold", zorder=11)
    txt_a = ax.text(INIT_A[0] - 0.20, INIT_A[1] + 0.17, "Init A", ha="left", va="center", **label_style)
    txt_b = ax.text(INIT_B[0] - 0.10, INIT_B[1] + 0.18, "Init B", ha="center", va="center", **label_style)
    for txt in (txt_a, txt_b):
        txt.set_path_effects([pe.withStroke(linewidth=3.4, foreground="black", alpha=0.38)])


def add_minimum_markers(ax) -> None:
    ax.scatter(*MINIMUM_A, marker="x", s=360, color="#3b1d0f", linewidth=5.0, zorder=12)
    ax.scatter(*MINIMUM_B, marker="x", s=360, color="#3b1d0f", linewidth=5.0, zorder=12)

    label_style = dict(color="white", fontsize=34, weight="bold", zorder=13)
    txt_a = ax.text(
        MINIMUM_A[0] - 0.22,
        MINIMUM_A[1] - 0.34,
        "$\\boldsymbol{\\theta}_A$",
        ha="left",
        va="center",
        **label_style,
    )
    txt_b = ax.text(
        MINIMUM_B[0] - 0.22,
        MINIMUM_B[1] - 0.34,
        "$\\boldsymbol{\\theta}_B$",
        ha="left",
        va="center",
        **label_style,
    )
    for txt in (txt_a, txt_b):
        txt.set_path_effects([pe.withStroke(linewidth=4.6, foreground="black", alpha=0.46)])


def add_three_minimum_markers(ax, *, include_permuted_a: bool = True) -> None:
    points = [MINIMUM_A, MINIMUM_B]
    labels = ["$\\boldsymbol{\\theta}_A$", "$\\boldsymbol{\\theta}_B$"]
    offsets = [np.array([-0.30, -0.32]), np.array([0.15, -0.42])]

    if include_permuted_a:
        points.append(PERMUTED_A)
        labels.append("$P(\\boldsymbol{\\theta}_A)$")
        offsets.append(np.array([-0.74, 0.34]))

    for point in points:
        ax.scatter(*point, marker="x", s=360, color="#3b1d0f", linewidth=5.0, zorder=12)

    label_style = dict(color="white", fontsize=34, weight="bold", zorder=13)
    for point, label, offset in zip(points, labels, offsets):
        txt = ax.text(
            *(point + offset),
            label,
            ha="left",
            va="center",
            **label_style,
        )
        txt.set_path_effects([pe.withStroke(linewidth=4.6, foreground="black", alpha=0.46)])


def sample_polyline(points: np.ndarray, samples_per_segment: int = 40) -> np.ndarray:
    pieces = []
    for start, end in zip(points[:-1], points[1:]):
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
        pieces.append(start * (1.0 - t) + end * t)
    return np.vstack([*pieces, points[-1:]])


def add_path_arrows(ax, points: np.ndarray, color: str, positions: tuple[int, ...]) -> None:
    sampled = sample_polyline(points, samples_per_segment=28)
    for idx in positions:
        start = sampled[idx]
        end = sampled[min(idx + 5, len(sampled) - 1)]
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", lw=0, color=color, mutation_scale=22),
            zorder=14,
        )


def add_optimization_paths(ax) -> None:
    for points, color, arrows in (
        (PATH_A, "#ffd33d", (38, 90, 136)),
        (PATH_B, "#ff8c00", (28, 70, 104)),
    ):
        curve = sample_polyline(points)
        ax.plot(
            curve[:, 0],
            curve[:, 1],
            color=color,
            linewidth=4.2,
            solid_capstyle="round",
            zorder=8,
            path_effects=[pe.Stroke(linewidth=6.4, foreground="black", alpha=0.22), pe.Normal()],
        )
        add_path_arrows(ax, points, color, arrows)


def add_linear_interpolation(ax) -> None:
    endpoints = np.vstack([MINIMUM_A, MINIMUM_B])
    ax.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        color="#d62728",
        linewidth=4.8,
        linestyle=(0, (1.4, 5.2)),
        dash_capstyle="round",
        zorder=9,
        path_effects=[pe.Stroke(linewidth=7.2, foreground="white", alpha=0.45), pe.Normal()],
    )


def add_permutation_arrow(ax) -> None:
    ax.scatter(*PERMUTED_B, marker="x", s=390, color="#1359c8", linewidth=5.0, zorder=15)
    txt = ax.text(
        PERMUTED_B[0] - 0.10,
        PERMUTED_B[1] + 0.34,
        "$P(\\boldsymbol{\\theta}_B)$",
        color="white",
        fontsize=29,
        weight="bold",
        ha="right",
        va="center",
        zorder=16,
    )
    txt.set_path_effects([pe.withStroke(linewidth=4.4, foreground="black", alpha=0.48)])

    ax.annotate(
        "",
        xy=PERMUTED_B + np.array([0.05, 0.05]),
        xytext=MINIMUM_B + np.array([-0.02, -0.02]),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1359c8",
            lw=4.8,
            mutation_scale=28,
            connectionstyle="arc3,rad=0.22",
            path_effects=[pe.Stroke(linewidth=7.2, foreground="white", alpha=0.58), pe.Normal()],
        ),
        zorder=14,
    )
    label = ax.text(
        0.10,
        0.17,
        "$P$",
        color="white",
        fontsize=27,
        weight="bold",
        ha="center",
        va="center",
        zorder=16,
    )
    label.set_path_effects([pe.withStroke(linewidth=3.8, foreground="black", alpha=0.45)])


def add_permuted_linear_interpolation(ax) -> None:
    endpoints = np.vstack([MINIMUM_A, PERMUTED_B])
    ax.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        color="#d62728",
        linewidth=6.2,
        linestyle=(0, (1.4, 5.2)),
        dash_capstyle="round",
        zorder=17,
        path_effects=[pe.Stroke(linewidth=9.0, foreground="white", alpha=0.58), pe.Normal()],
    )


def bezier_path(start: np.ndarray, control: np.ndarray, end: np.ndarray, samples: int = 180) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples)[:, None]
    return (1.0 - t) ** 2 * start + 2.0 * (1.0 - t) * t * control + t**2 * end


def add_curved_connectivity_paths(
    ax,
    *,
    include_ab: bool = True,
    include_permuted_a: bool = True,
) -> None:
    paths = []
    if include_ab:
        paths.append((curved_path_ab(), 112))
    if include_permuted_a:
        paths.append((curved_path_pa(), 100))

    for path, arrow_idx in paths:
        ax.plot(
            path[:, 0],
            path[:, 1],
            color="#7ed6ff",
            linewidth=5.8,
            solid_capstyle="round",
            zorder=11,
            path_effects=[pe.Stroke(linewidth=8.8, foreground="white", alpha=0.58), pe.Normal()],
        )
        ax.annotate(
            "",
            xy=path[min(arrow_idx + 7, len(path) - 1)],
            xytext=path[arrow_idx],
            arrowprops=dict(arrowstyle="-|>", lw=0, color="#7ed6ff", mutation_scale=27),
            zorder=14,
        )


def save_three_minima_variant(
    out_path: Path,
    *,
    include_ab_path: bool = False,
    include_permuted_a: bool = False,
    include_permuted_a_path: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 10.1), dpi=180)
    filled = draw_three_minima_base(
        ax,
        include_permuted_a=include_permuted_a,
        include_ab_channel=include_ab_path,
        include_permuted_a_channel=include_permuted_a_path,
    )
    if include_ab_path or include_permuted_a_path:
        add_curved_connectivity_paths(
            ax,
            include_ab=include_ab_path,
            include_permuted_a=include_permuted_a_path,
        )
    add_three_minimum_markers(ax, include_permuted_a=include_permuted_a)
    add_colorbar(fig, ax, filled)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(out_path)


def save_three_minima_curved_variant() -> None:
    save_three_minima_variant(
        OUT_THREE_MINIMA_STAGE0,
        include_ab_path=False,
        include_permuted_a=False,
        include_permuted_a_path=False,
    )
    save_three_minima_variant(
        OUT_THREE_MINIMA_STAGE1,
        include_ab_path=True,
        include_permuted_a=False,
        include_permuted_a_path=False,
    )
    save_three_minima_variant(
        OUT_THREE_MINIMA_STAGE2,
        include_ab_path=True,
        include_permuted_a=True,
        include_permuted_a_path=False,
    )
    save_three_minima_variant(
        OUT_THREE_MINIMA_STAGE3,
        include_ab_path=True,
        include_permuted_a=True,
        include_permuted_a_path=True,
    )
    save_three_minima_variant(
        OUT_THREE_MINIMA_CURVED,
        include_ab_path=True,
        include_permuted_a=True,
        include_permuted_a_path=True,
    )


def save_variant(
    out_path: Path,
    *,
    include_inits: bool = False,
    include_minima: bool = False,
    include_training_paths: bool = False,
    include_linear_path: bool = False,
    include_permutation: bool = False,
    include_permuted_linear_path: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 10.1), dpi=180)
    filled = draw_base(ax)
    if include_training_paths:
        add_optimization_paths(ax)
    if include_linear_path:
        add_linear_interpolation(ax)
    if include_permuted_linear_path:
        add_permuted_linear_interpolation(ax)
    if include_inits:
        add_init_markers(ax)
    if include_minima:
        add_minimum_markers(ax)
    if include_permutation:
        add_permutation_arrow(ax)
    add_colorbar(fig, ax, filled)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(out_path)


def main() -> None:
    OUT_CLEAN.parent.mkdir(parents=True, exist_ok=True)
    save_variant(OUT_CLEAN)
    save_variant(OUT_INITS, include_inits=True)
    save_variant(OUT_MINIMA_ONLY, include_minima=True)
    save_variant(
        OUT_FULL,
        include_inits=True,
        include_minima=True,
        include_training_paths=True,
    )
    save_variant(
        OUT_LINEAR_ONLY,
        include_inits=True,
        include_minima=True,
        include_training_paths=True,
        include_linear_path=True,
    )
    save_variant(
        OUT_PERMUTATION,
        include_minima=True,
        include_permutation=True,
    )
    save_variant(
        OUT_PERMUTATION_LINEAR,
        include_minima=True,
        include_permutation=True,
        include_permuted_linear_path=True,
    )
    save_three_minima_curved_variant()


if __name__ == "__main__":
    main()
