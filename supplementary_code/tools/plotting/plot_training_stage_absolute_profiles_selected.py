#!/usr/bin/env python3
"""Plot selected training-stage interpolation profiles from an existing report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


METHODS = {
    "raw": ("raw", "tab:blue"),
    "wm": ("wm", "tab:orange"),
    "base": ("sinkhorn", "tab:green"),
    "scale": ("scale", "tab:purple"),
}
SUBSETS = ("train_eval", "test_eval")
METRICS = (("losses", "Cross-entropy loss"), ("errors", "Classification error (%)"))


def plot(result_root: Path, output: Path) -> None:
    profiles_path = result_root / "report" / "profiles.json"
    profiles = json.loads(profiles_path.read_text())
    selected = [row for row in profiles if row["method"] in METHODS]
    if not selected:
        raise ValueError(f"No selected methods found in {profiles_path}")

    replicates = sorted({row["replicate"] for row in selected})
    pairs = sorted({tuple(row["epochs"]) for row in selected})
    index = {
        (row["replicate"], tuple(row["epochs"]), row["method"], row["subset"]): row
        for row in selected
    }

    expected = len(replicates) * len(pairs) * len(METHODS) * len(SUBSETS)
    if len(index) != expected:
        raise ValueError(
            f"Expected {expected} unique profiles, found {len(index)} in {profiles_path}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for replicate in replicates:
            for epoch_a, epoch_b in pairs:
                fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
                legend_handles = []
                for row_index, subset in enumerate(SUBSETS):
                    for column_index, (metric, ylabel) in enumerate(METRICS):
                        axis = axes[row_index, column_index]
                        for method, (label, color) in METHODS.items():
                            row = index[
                                (replicate, (epoch_a, epoch_b), method, subset)
                            ]
                            (line,) = axis.plot(
                                row["alphas"],
                                row[metric],
                                label=label,
                                color=color,
                                linewidth=1.8,
                            )
                            if row_index == 0 and column_index == 0:
                                legend_handles.append(line)
                        axis.set_title(
                            f"{'Train' if subset == 'train_eval' else 'Test'} · "
                            f"{'loss' if metric == 'losses' else 'error'}"
                        )
                        axis.set_xlabel("Interpolation coefficient α")
                        axis.set_ylabel(ylabel)
                        axis.grid(alpha=0.2)

                fig.legend(
                    handles=legend_handles,
                    labels=[handle.get_label() for handle in legend_handles],
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.94),
                    ncol=len(METHODS),
                    frameon=True,
                )
                fig.suptitle(
                    f"Seed pair {replicate}: completed epoch {epoch_a} → {epoch_b}",
                    y=0.985,
                )
                fig.tight_layout(rect=(0, 0, 1, 0.90))
                pdf.savefig(fig)
                plt.close(fig)

    print(f"Wrote {output} ({len(replicates) * len(pairs)} pages)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or (
        args.result_root / "report" / "absolute_profiles_raw_wm_sinkhorn_scale.pdf"
    )
    plot(args.result_root, output)


if __name__ == "__main__":
    main()
