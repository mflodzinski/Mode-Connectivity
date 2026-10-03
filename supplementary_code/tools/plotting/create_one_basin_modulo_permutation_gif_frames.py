"""Create annotated GIF frames for the one-basin modulo permutation slide."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path as MplPath
import matplotlib.pyplot as plt
import numpy as np

from create_pseudo_loss_landscape_clean_variants import (
    add_colorbar,
    draw_base,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "thesis" / "tex" / "one_basin_modulo_permutation_gif_frames" / "annotated"

# Put the representatives on the peripheries of their low-loss basins rather
# than at the basin centers. The permuted representative is placed on the same
# basin as theta_A so their straight interpolation remains low loss.
THETA_A = np.array([-0.93, -0.66])
THETA_B = np.array([0.94, 0.58])
PERMUTED_THETA_B = np.array([-1.36, -1.12])


def add_note_arrow(
    ax,
    text: str,
    text_xy: tuple[float, float],
    target_xy: tuple[float, float],
    *,
    color: str,
    ha: str = "center",
    rad: float = 0.0,
) -> None:
    """Add a readable text label with an arrow to a feature on the plot."""
    txt = ax.text(
        *text_xy,
        text,
        color=color,
        fontsize=25,
        weight="bold",
        ha=ha,
        va="center",
        zorder=30,
    )
    txt.set_path_effects([pe.withStroke(linewidth=5.2, foreground="white", alpha=0.92)])
    ax.annotate(
        "",
        xy=target_xy,
        xytext=text_xy,
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=4.8,
            mutation_scale=28,
            connectionstyle=f"arc3,rad={rad}",
            path_effects=[pe.Stroke(linewidth=7.2, foreground="white", alpha=0.80), pe.Normal()],
        ),
        zorder=29,
    )


def add_high_loss_label(ax) -> None:
    add_note_arrow(
        ax,
        "High loss region",
        (0.86, -1.42),
        (0.02, 0.02),
        color="#00a36c",
        ha="center",
        rad=0.20,
    )


def add_low_loss_label(ax) -> None:
    target = tuple((THETA_A + PERMUTED_THETA_B) / 2.0)
    add_note_arrow(
        ax,
        "Low loss interpolation",
        (-1.08, -1.66),
        target,
        color="#00a36c",
        ha="center",
        rad=-0.18,
    )


def add_theta_markers(ax, *, include_permuted: bool = False) -> None:
    points = [THETA_A, THETA_B]
    labels = ["$\\boldsymbol{\\theta}_A$", "$\\boldsymbol{\\theta}_B$"]
    offsets = [np.array([0.06, -0.22]), np.array([0.06, -0.24])]
    colors = ["#ff8c00", "#ff8c00"]

    if include_permuted:
        points.append(PERMUTED_THETA_B)
        labels.append("$P(\\boldsymbol{\\theta}_B)$")
        offsets.append(np.array([-0.68, 0.06]))
        colors.append("#ff8c00")

    for point, color in zip(points, colors):
        ax.scatter(*point, marker="x", s=360, color=color, linewidth=5.0, zorder=25)

    label_style = dict(color="white", fontsize=34, weight="bold", zorder=26)
    for point, label, offset in zip(points, labels, offsets):
        txt = ax.text(*(point + offset), label, ha="left", va="center", **label_style)
        txt.set_path_effects([pe.withStroke(linewidth=4.6, foreground="black", alpha=0.46)])


def add_linear_interpolation_path(ax) -> None:
    endpoints = np.vstack([THETA_A, THETA_B])
    ax.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        color="#d62728",
        linewidth=4.8,
        linestyle=(0, (2.8, 2.0)),
        dash_capstyle="round",
        zorder=9,
        path_effects=[pe.Stroke(linewidth=7.2, foreground="white", alpha=0.45), pe.Normal()],
    )


def add_permutation(ax) -> None:
    path = MplPath(
        [
            THETA_B + np.array([-0.01, 0.05]),
            np.array([0.55, 1.55]),
            np.array([-1.40, 1.18]),
            PERMUTED_THETA_B + np.array([0.05, 0.08]),
        ],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    arrow = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        color="#1359c8",
        lw=4.8,
        mutation_scale=28,
        zorder=17,
    )
    arrow.set_path_effects([pe.Stroke(linewidth=7.2, foreground="white", alpha=0.58), pe.Normal()])
    ax.add_patch(arrow)

    label = ax.text(
        -0.24,
        1.12,
        "$P$",
        color="white",
        fontsize=27,
        weight="bold",
        ha="center",
        va="center",
        zorder=18,
    )
    label.set_path_effects([pe.withStroke(linewidth=3.8, foreground="black", alpha=0.45)])


def add_low_loss_interpolation_path(ax) -> None:
    endpoints = np.vstack([THETA_A, PERMUTED_THETA_B])
    ax.plot(
        endpoints[:, 0],
        endpoints[:, 1],
        color="#d62728",
        linewidth=6.2,
        linestyle=(0, (2.8, 2.0)),
        dash_capstyle="round",
        zorder=23,
        path_effects=[pe.Stroke(linewidth=9.0, foreground="white", alpha=0.58), pe.Normal()],
    )


def save_frame(
    out_path: Path,
    *,
    linear_path: bool = False,
    high_loss_notes: bool = False,
    permutation: bool = False,
    permuted_linear_path: bool = False,
    low_loss_note: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 10.1), dpi=180)
    filled = draw_base(ax)

    if linear_path:
        add_linear_interpolation_path(ax)

    if permutation:
        add_permutation(ax)
    if permuted_linear_path:
        add_low_loss_interpolation_path(ax)

    add_theta_markers(ax, include_permuted=permutation)
    if high_loss_notes:
        add_high_loss_label(ax)
    if low_loss_note:
        add_low_loss_label(ax)

    add_colorbar(fig, ax, filled)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(out_path)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_frame(OUT_DIR / "01_apparent_separate_minima.png")
    save_frame(
        OUT_DIR / "02_high_loss_linear_interpolation.png",
        linear_path=True,
        high_loss_notes=True,
    )
    save_frame(
        OUT_DIR / "02_high_loss_linear_interpolation_no_note.png",
        linear_path=True,
    )
    save_frame(
        OUT_DIR / "03_permutation_to_equivalent_representative.png",
        permutation=True,
    )
    save_frame(
        OUT_DIR / "04_low_loss_interpolation_after_permutation.png",
        permutation=True,
        permuted_linear_path=True,
        low_loss_note=True,
    )
    save_frame(
        OUT_DIR / "04_low_loss_interpolation_after_permutation_no_note.png",
        permutation=True,
        permuted_linear_path=True,
    )


if __name__ == "__main__":
    main()
