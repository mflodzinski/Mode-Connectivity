"""Annotate the existing pseudo loss landscape PNG with barrier paths.

This script intentionally uses plots/pseudo_loss_landscape_two_minima/pseudo_loss_landscape_two_minima.png as
the background image instead of recreating the landscape.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as pe
from scipy.ndimage import binary_dilation, convolve


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "plots" / "pseudo_loss_landscape_two_minima"
SOURCE = OUT_DIR / "pseudo_loss_landscape_two_minima.png"
OUT = OUT_DIR / "pseudo_loss_landscape_two_minima_high_barrier_paths.png"
OUT_LINEAR_ONLY = OUT_DIR / "pseudo_loss_landscape_two_minima_high_barrier_linear_only.png"


def cubic_bezier(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    n: int = 220,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, n)
    x = (1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
    return x, y


def remove_white_text(image: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    """Remove baked-in white labels by diffusing nearby background colors."""
    x0, y0, w, h = rect
    patch = image[y0 : y0 + h, x0 : x0 + w, :].copy()
    rgb = patch[..., :3]

    # Capture the white letters plus a few pixels around them, while avoiding
    # broad rectangular edits that make the contour plot look blurred.
    text_mask = (rgb[..., 0] > 0.62) & (rgb[..., 1] > 0.62) & (rgb[..., 2] > 0.62)
    text_mask = binary_dilation(text_mask, iterations=10)

    kernel = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    filled = patch.copy()
    remaining = text_mask.copy()
    for _ in range(120):
        known = (~remaining).astype(float)
        denom = convolve(known, kernel, mode="nearest")
        fillable = remaining & (denom > 0)
        if not np.any(fillable):
            break
        for channel in range(3):
            numer = convolve(filled[..., channel] * known, kernel, mode="nearest")
            update = numer / np.maximum(denom, 1e-8)
            filled[..., channel][fillable] = update[fillable]
        remaining[fillable] = False
        if not np.any(remaining):
            break

    image[y0 : y0 + h, x0 : x0 + w, :] = filled


def make_figure(out_path: Path, include_nonlinear_path: bool) -> None:
    image = mpimg.imread(SOURCE)
    height, width = image.shape[:2]

    # Remove the original minimum captions from the background before redrawing
    # them in positions that do not overlap the added paths.
    old_caption_boxes = [
        (520, 930, 370, 92),
        (1120, 510, 360, 86),
    ]
    for box in old_caption_boxes:
        remove_white_text(image, box)

    fig = plt.figure(figsize=(width / 180, height / 180), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    # Pixel coordinates in the original PNG, chosen to align with the existing
    # minimum markers. The image includes margins and the colorbar.
    minimum_a = (575, 920)
    minimum_b = (1228, 506)
    barrier_center = (905, 720)

    yy, xx = np.mgrid[0:height, 0:width]
    sigma_x = 250.0
    sigma_y = 205.0
    alpha = 0.48 * np.exp(-(((xx - barrier_center[0]) / sigma_x) ** 2 + ((yy - barrier_center[1]) / sigma_y) ** 2))

    # Restrict the highlight to the main plotting panel so the colorbar is not
    # recolored. This preserves the original figure layout.
    panel_mask = (xx > 105) & (xx < 1665) & (yy > 45) & (yy < 1462)
    alpha *= panel_mask

    yellow = np.zeros((height, width, 4), dtype=float)
    yellow[..., 0] = 1.00
    yellow[..., 1] = 0.92
    yellow[..., 2] = 0.00
    yellow[..., 3] = alpha
    ax.imshow(yellow)

    line_kwargs = dict(
        linewidth=7.0,
        linestyle=(0, (1.0, 1.65)),
        dash_capstyle="round",
        solid_capstyle="round",
        zorder=20,
    )

    red_x = np.linspace(minimum_a[0], minimum_b[0], 160)
    red_y = np.linspace(minimum_a[1], minimum_b[1], 160)
    ax.plot(
        red_x[8:-8],
        red_y[8:-8],
        color="#e63946",
        **line_kwargs,
    )

    if include_nonlinear_path:
        curve_x, curve_y = cubic_bezier(
            minimum_a,
            (610, 1225),
            (1165, 1195),
            minimum_b,
        )
        ax.plot(curve_x[10:-10], curve_y[10:-10], color="#78d7ff", **line_kwargs)

    caption_style = dict(color="white", fontsize=20, weight="bold", zorder=30)
    txt_a = ax.text(360, 1000, "Minimum A", ha="left", va="center", **caption_style)
    txt_b = ax.text(1260, 575, "Minimum B", ha="left", va="center", **caption_style)
    for txt in (txt_a, txt_b):
        txt.set_path_effects([pe.withStroke(linewidth=3.0, foreground="black", alpha=0.35)])

    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(out_path)


def main() -> None:
    make_figure(OUT, include_nonlinear_path=True)
    make_figure(OUT_LINEAR_ONLY, include_nonlinear_path=False)


if __name__ == "__main__":
    main()
