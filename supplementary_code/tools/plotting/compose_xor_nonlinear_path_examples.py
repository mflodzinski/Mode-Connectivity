"""Compose the width-2/width-3 nonlinear XOR examples for the paper.

The source panels are cropped from the original experiment figures so the
decision-boundary colours and contours remain unchanged.  This script only
standardizes the surrounding labels and layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURES = ROOT / "weekly_thesis_update(4)" / "paper_aistats2027" / "figures"

# Pixel bounds of the five square plotting panels in the original 4578x1002
# exports.  Labels outside these bounds are deliberately replaced below.
PANEL_X_BOUNDS = (
    (227, 1042),
    (1104, 1919),
    (1981, 2796),
    (2858, 3673),
    (3735, 4550),
)
PANEL_Y_BOUNDS = (94, 909)


def crop_panels(path: Path) -> list[Image.Image]:
    image = Image.open(path).convert("RGB")
    y0, y1 = PANEL_Y_BOUNDS
    return [image.crop((x0, y0, x1, y1)) for x0, x1 in PANEL_X_BOUNDS]


def compose(width2_path: Path, width3_path: Path, output_path: Path) -> None:
    rows = (
        ("Hidden size 2", crop_panels(width2_path)),
        ("Hidden size 3", crop_panels(width3_path)),
    )
    lambdas = (0, 0.25, 0.5, 0.75, 1)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6.7))
    for row_index, (_, panels) in enumerate(rows):
        for column_index, (axis, panel) in enumerate(zip(axes[row_index], panels)):
            axis.imshow(panel, interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

            if row_index == 0:
                axis.set_title(
                    rf"$\lambda = {lambdas[column_index]:g}$",
                    fontsize=22,
                    fontweight="normal",
                    pad=10,
                )
            if row_index == 1:
                axis.set_xlabel(r"$x_1$", fontsize=24, labelpad=7)
            if column_index == 0:
                axis.set_ylabel(r"$x_2$", fontsize=24, labelpad=7)

        row_center = 0.73 if row_index == 0 else 0.285
        fig.text(
            0.018,
            row_center,
            rows[row_index][0],
            rotation=90,
            va="center",
            ha="center",
            fontsize=24,
            fontweight="bold",
        )

    fig.subplots_adjust(
        left=0.058,
        right=0.995,
        bottom=0.08,
        top=0.93,
        wspace=0.08,
        hspace=0.12,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--width2",
        type=Path,
        default=PAPER_FIGURES / "unsuccesful_10-14_bezier_3bend.png",
    )
    parser.add_argument(
        "--width3",
        type=Path,
        default=PAPER_FIGURES / "3hidden_xor_bezier.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PAPER_FIGURES / "nonlinear_xor_paths_hidden_sizes.png",
    )
    args = parser.parse_args()
    compose(args.width2, args.width3, args.output)


if __name__ == "__main__":
    main()
