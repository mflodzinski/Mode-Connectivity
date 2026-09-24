#!/usr/bin/env python3
"""Plot pointwise loss excess profiles for a training-stage alignment grid."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


METHOD_ORDER = ("raw", "base", "scale")
METHOD_LABELS = {
    "raw": "Raw",
    "base": "Sinkhorn",
    "scale": "Sinkhorn + scale",
}


def pointwise_profiles(record: dict) -> tuple[list[float], list[float]]:
    alphas = record["alphas"]
    losses = record["losses"]
    left, right = losses[0], losses[-1]
    chord = [(1.0 - alpha) * left + alpha * right for alpha in alphas]
    chord_excess = [loss - baseline for loss, baseline in zip(losses, chord)]
    worse = max(left, right)
    worse_excess = [loss - worse for loss in losses]
    return chord_excess, worse_excess


def plot(report_dir: Path, output: Path) -> int:
    records = json.loads((report_dir / "profiles.json").read_text())
    records = [record for record in records if record["subset"] == "test_eval"]
    index = {
        (record["replicate"], *record["epochs"], record["method"]): record
        for record in records
    }
    pairs = sorted(
        {(record["replicate"], *record["epochs"]) for record in records}
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output) as pdf:
        for replicate, left_epoch, right_epoch in pairs:
            fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.0), sharex=True)
            for method in METHOD_ORDER:
                record = index[(replicate, left_epoch, right_epoch, method)]
                chord_excess, worse_excess = pointwise_profiles(record)
                barrier = record["loss"]
                label = (
                    f"{METHOD_LABELS[method]} "
                    f"($B_{{chord}}$={barrier['chord']:.3f}, "
                    f"$B_{{worse}}$={barrier['worse']:.3f})"
                )
                axes[0].plot(record["alphas"], record["losses"], label=label)
                axes[1].plot(record["alphas"], chord_excess, label=METHOD_LABELS[method])
                axes[2].plot(record["alphas"], worse_excess, label=METHOD_LABELS[method])

            axes[0].set_ylabel("Test cross-entropy")
            axes[0].set_title("Absolute loss")
            axes[0].legend(fontsize=8)
            axes[1].set_ylabel("Loss minus endpoint chord")
            axes[1].set_title("Pointwise chord excess")
            axes[2].set_ylabel("Loss minus worse endpoint")
            axes[2].set_title("Pointwise worse-endpoint excess")
            axes[2].set_xlabel("Interpolation coefficient $\\alpha$")
            for axis in axes:
                axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
                axis.grid(alpha=0.2)
            fig.suptitle(
                f"Replicate {replicate}: completed epochs "
                f"{left_epoch} $\\rightarrow$ {right_epoch}"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
            plt.close(fig)
    return len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report_dir",
        type=Path,
        help="Directory containing profiles.json from alignment-grid reporting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PDF (default: REPORT_DIR/test_loss_barrier_profiles.pdf).",
    )
    args = parser.parse_args()
    output = args.output or args.report_dir / "test_loss_barrier_profiles.pdf"
    pages = plot(args.report_dir, output)
    print(f"Wrote {output} ({pages} pages)")


if __name__ == "__main__":
    main()
