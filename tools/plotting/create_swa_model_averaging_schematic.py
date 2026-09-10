"""Create schematics of SWA-style model averaging on train/test-loss landscapes."""

from __future__ import annotations

from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAIN_OUT_PATH = ROOT / "plots" / "swa_model_averaging_train_loss.png"
TEST_OUT_PATH = ROOT / "plots" / "swa_model_averaging_test_loss.png"


def train_loss_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    broad_basin = 0.11 * (x**2 + 0.70 * y**2)
    flat_center = -1.10 * np.exp(-((x / 1.35) ** 2 + (y / 1.00) ** 2))
    narrow_well = -0.32 * np.exp(-(((x + 0.78) / 0.34) ** 2 + ((y - 0.35) / 0.28) ** 2))
    small_ripples = 0.035 * np.sin(4.2 * x) * np.cos(3.0 * y)
    return broad_basin + flat_center + narrow_well + small_ripples + 1.35


def test_loss_surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lower_flat_center = -0.34 * np.exp(-((x / 0.82) ** 2 + (y / 0.62) ** 2))
    return train_loss_surface(x, y) + lower_flat_center


def sample_curve(points: np.ndarray, samples_per_segment: int = 36) -> np.ndarray:
    pieces = []
    for start, end in zip(points[:-1], points[1:]):
        t = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)[:, None]
        smooth_t = 3 * t**2 - 2 * t**3
        pieces.append(start * (1.0 - smooth_t) + end * smooth_t)
    return np.vstack([*pieces, points[-1:]])


def add_label(ax, xy: tuple[float, float], text: str, *, fontsize: int = 18, ha: str = "center") -> None:
    label = ax.text(
        xy[0],
        xy[1],
        text,
        color="white",
        fontsize=fontsize,
        weight="bold",
        ha=ha,
        va="center",
        zorder=20,
    )
    label.set_path_effects([pe.withStroke(linewidth=4.0, foreground="black", alpha=0.45)])


def render_landscape(
    out_path: Path,
    zz: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    levels: np.ndarray,
    colorbar_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(13.0, 7.3), dpi=180)
    filled = ax.contourf(xx, yy, zz, levels=levels, cmap="viridis", extend="both")
    ax.contour(xx, yy, zz, levels=levels, colors="black", linewidths=0.45, alpha=0.30)

    # A cyclic-LR trajectory after the model has already reached a low-loss basin.
    trajectory_nodes = np.array(
        [
            [-0.92, 0.36],
            [-0.20, 0.48],
            [0.52, 0.14],
            [0.30, -0.34],
            [-0.34, -0.28],
        ]
    )
    trajectory = sample_curve(trajectory_nodes)
    checkpoints = trajectory_nodes[[1, 2, 3, 4]]
    avg_point = checkpoints.mean(axis=0)
    trained_model = trajectory_nodes[0]

    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#ffb000",
        linewidth=4.2,
        zorder=9,
        solid_capstyle="round",
        path_effects=[pe.Stroke(linewidth=6.8, foreground="black", alpha=0.22), pe.Normal()],
    )
    for idx in (26, 74, 122):
        ax.annotate(
            "",
            xy=trajectory[min(idx + 8, len(trajectory) - 1)],
            xytext=trajectory[idx],
            arrowprops=dict(arrowstyle="-|>", lw=0, color="#ffb000", mutation_scale=22),
            zorder=12,
        )

    ax.scatter(
        checkpoints[:, 0],
        checkpoints[:, 1],
        s=150,
        color="#1f77b4",
        edgecolor="white",
        linewidth=1.8,
        zorder=15,
        label="Saved checkpoints",
    )
    ax.scatter(
        *trained_model,
        marker="*",
        s=520,
        color="#ffd33d",
        edgecolor="black",
        linewidth=1.6,
        zorder=16,
    )
    ax.scatter(
        *avg_point,
        marker="o",
        s=210,
        color="#d62728",
        edgecolor="white",
        linewidth=2.0,
        zorder=18,
    )

    add_label(ax, (-1.72, 0.92), "trained model", fontsize=17, ha="left")
    ax.annotate(
        "",
        xy=trained_model + np.array([0.03, 0.02]),
        xytext=(-1.20, 0.82),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2.4, mutation_scale=18),
        zorder=14,
    )
    add_label(ax, (0.50, -1.05), "average model", fontsize=18, ha="left")
    ax.annotate(
        "",
        xy=avg_point + np.array([0.02, -0.01]),
        xytext=(0.47, -0.88),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=2.4, mutation_scale=18),
        zorder=14,
    )

    ax.set_xlim(-2.45, 2.45)
    ax.set_ylim(-1.85, 1.75)
    ax.set_xlabel("$u_1$", fontsize=25)
    ax.set_ylabel("$u_2$", fontsize=25)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)

    cbar = fig.colorbar(filled, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_ticks([filled.levels[0], filled.levels[-1]])
    cbar.set_ticklabels(["low", "high"])
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label(colorbar_label, fontsize=20, weight="bold", labelpad=12)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(out_path)


def main() -> None:
    TRAIN_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-2.6, 2.6, 500)
    y = np.linspace(-2.0, 2.0, 420)
    xx, yy = np.meshgrid(x, y)
    train_zz = train_loss_surface(xx, yy)
    test_zz = test_loss_surface(xx, yy)
    levels = np.linspace(np.percentile(train_zz, 0.5), np.percentile(train_zz, 99.0), 40)

    render_landscape(TRAIN_OUT_PATH, train_zz, xx, yy, levels, "TRAIN LOSS")
    render_landscape(TEST_OUT_PATH, test_zz, xx, yy, levels, "TEST LOSS")


if __name__ == "__main__":
    main()
