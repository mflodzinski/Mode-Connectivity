from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "thesis" / "figures" / "new"
SOURCE = OUT_DIR / "construction.png"

# Pixel coordinates in the original construction.png.
POINTS = {
    "init": np.array([535.0, 183.0]),
    "split": np.array([535.0, 291.0]),
    "theta0": np.array([375.0, 474.0]),
    "theta1": np.array([613.0, 471.0]),
    "theta1_prime": np.array([1032.0, 494.0]),
    "p_theta1_prime": np.array([397.0, 609.0]),
}

# Plot rectangle detected from the black border of construction.png.
AXES_BOUNDS = (90, 24, 1361, 934)  # left, top, right, bottom


def load_clean_background() -> np.ndarray:
    """Use the original frame/colorbar with a clean two-basin contour interior."""
    image = np.array(Image.open(SOURCE).convert("RGB"))
    left, top, right, bottom = AXES_BOUNDS

    cleaned = image.copy()
    model = contour_model(image.shape[:2])
    cleaned[top + 2 : bottom, left + 2 : right] = model[top + 2 : bottom, left + 2 : right]
    return cleaned


def contour_model(shape: tuple[int, int]) -> np.ndarray:
    """Smooth contour colors matching the original figure's two-basin geometry."""
    height, width = shape
    yy, xx = np.indices((height, width))
    left, top, right, bottom = AXES_BOUNDS

    left_center = np.array([455.0, 493.0])
    right_center = np.array([1035.0, 493.0])
    x_radius = 365.0
    y_radius = 385.0
    left_loss = ((xx - left_center[0]) / x_radius) ** 2 + ((yy - left_center[1]) / y_radius) ** 2
    right_loss = ((xx - right_center[0]) / x_radius) ** 2 + ((yy - right_center[1]) / y_radius) ** 2
    tau = 0.32
    loss = -tau * np.log(np.exp(-left_loss / tau) + np.exp(-right_loss / tau))
    loss += 0.12 * ((yy - (top + bottom) / 2.0) / y_radius) ** 4
    loss = np.clip(loss / 1.28, 0.0, 1.0)

    levels = np.linspace(0.0, 1.0, 32)
    quantized = levels[np.searchsorted(levels, loss, side="right").clip(max=len(levels) - 1)]
    cmap = plt.get_cmap("viridis")
    rgb = (cmap(quantized)[..., :3] * 255).astype(np.uint8)
    return rgb


def draw_point(ax, key: str, label: str, offset: tuple[float, float]):
    x, y = POINTS[key]
    ax.scatter([x], [y], s=125, facecolor="white", edgecolor="none", zorder=8)
    ax.text(
        x + offset[0],
        y + offset[1],
        label,
        color="white",
        fontsize=24,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=9,
    )


def draw_arrow(ax, start, end, *, rad=0.0, lw=3.4, ms=22):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=ms,
        connectionstyle=f"arc3,rad={rad}",
        linewidth=lw,
        color="white",
        shrinkA=16,
        shrinkB=16,
        zorder=6,
    )
    ax.add_patch(arrow)


def draw_stage(stage: int, output_name: str, background: np.ndarray):
    h, w = background.shape[:2]
    fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(background)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    draw_point(ax, "init", r"$\theta_{init}$", offset=(-24, -30))

    if stage >= 2:
        draw_point(ax, "split", r"$\theta_{split}$", offset=(18, 0))
        draw_arrow(ax, POINTS["init"], POINTS["split"])

    if stage >= 3:
        draw_point(ax, "theta0", r"$\theta_0$", offset=(-56, -24))
        draw_point(ax, "theta1", r"$\theta_1$", offset=(18, 12))
        draw_arrow(ax, POINTS["split"], POINTS["theta0"])
        draw_arrow(ax, POINTS["split"], POINTS["theta1"])
        ax.plot(
            [POINTS["theta0"][0], POINTS["theta1"][0]],
            [POINTS["theta0"][1], POINTS["theta1"][1]],
            color="#20c8ff",
            linestyle=":",
            linewidth=3.8,
            zorder=5,
        )

    if stage >= 4:
        draw_point(ax, "theta1_prime", r"$\theta_1'$", offset=(-2, -32))
        draw_arrow(ax, POINTS["theta1"], POINTS["theta1_prime"], rad=-0.35)

    if stage >= 5:
        draw_point(ax, "p_theta1_prime", r"$P(\theta_1')$", offset=(-82, 38))
        draw_arrow(ax, POINTS["theta1_prime"], POINTS["p_theta1_prime"], rad=-0.34)
        ax.plot(
            [POINTS["theta0"][0], POINTS["p_theta1_prime"][0]],
            [POINTS["theta0"][1], POINTS["p_theta1_prime"][1]],
            color="#70d13b",
            linestyle=":",
            linewidth=3.8,
            zorder=5,
        )

    fig.savefig(OUT_DIR / output_name, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    background = load_clean_background()
    stages = [
        (1, "construction_stage_1_theta_init.png"),
        (2, "construction_stage_2_theta_split.png"),
        (3, "construction_stage_3_theta0_theta1_interpolation.png"),
        (4, "construction_stage_4_theta1_prime.png"),
        (5, "construction_stage_5_recovered_interpolation.png"),
    ]
    for stage, output_name in stages:
        draw_stage(stage, output_name, background)


if __name__ == "__main__":
    main()
