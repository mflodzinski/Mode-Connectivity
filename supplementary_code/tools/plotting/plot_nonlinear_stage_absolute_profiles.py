#!/usr/bin/env python3
"""Plot every saved nonlinear-stage absolute interpolation profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


GROUPS = {
    "same_stage": {"same"},
    "cross_stage": {"final_left", "final_right"},
    "within_run": {"within_a", "within_b"},
}
KIND_ORDER = {
    "same": 0,
    "final_left": 1,
    "final_right": 2,
    "within_a": 3,
    "within_b": 4,
}
COLORS = {"linear": "tab:orange", "curve": "tab:blue"}


def method_label(record: dict, metric: str) -> str:
    name = "linear" if record["method"] == "linear" else record["family"]
    barrier = record["profile"][metric]
    return (
        f"{name}: chord={barrier['chord']:.3g}, "
        f"worse={barrier['worse']:.3g}"
    )


def pair_title(pair: dict) -> str:
    return (
        f"replicate {pair['replicate']} · {pair['kind']} · "
        f"seed {pair['left_seed']}@{pair['left_epoch']} → "
        f"seed {pair['right_seed']}@{pair['right_epoch']}"
    )


def write_group(records: list[dict], kinds: set[str], output: Path) -> int:
    pair_records: dict[str, list[dict]] = {}
    pairs: dict[str, dict] = {}
    for record in records:
        pair = record["pair"]
        if pair["kind"] not in kinds or record["subset"] == "validation_audit":
            continue
        pair_records.setdefault(pair["id"], []).append(record)
        pairs[pair["id"]] = pair

    ordered = sorted(
        pairs.values(),
        key=lambda pair: (
            pair["replicate"],
            KIND_ORDER[pair["kind"]],
            pair["n"],
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        for pair in ordered:
            index = {
                (record["subset"], record["method"]): record
                for record in pair_records[pair["id"]]
            }
            expected = {
                (subset, method)
                for subset in ("train_eval", "test_eval")
                for method in ("linear", "curve")
            }
            if set(index) != expected:
                raise ValueError(
                    f"Incomplete profiles for {pair['id']}: {sorted(index)}"
                )

            fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.7), sharex=True)
            for row_index, subset in enumerate(("train_eval", "test_eval")):
                for column_index, (metric, values_key, ylabel) in enumerate(
                    (
                        ("loss", "losses", "Cross-entropy loss"),
                        ("error", "errors", "Classification error (%)"),
                    )
                ):
                    axis = axes[row_index, column_index]
                    for method in ("linear", "curve"):
                        record = index[(subset, method)]
                        profile = record["profile"]
                        axis.plot(
                            profile["alphas"],
                            profile[values_key],
                            color=COLORS[method],
                            linewidth=1.8,
                            label=method_label(record, metric),
                        )
                    axis.set(
                        title=(
                            f"{'Train' if subset == 'train_eval' else 'Test'} · "
                            f"{'loss' if metric == 'loss' else 'error'}"
                        ),
                        xlabel="Path coefficient t",
                        ylabel=ylabel,
                    )
                    axis.grid(alpha=0.2)
                    axis.legend(fontsize=8)
            fig.suptitle(pair_title(pair), y=0.99)
            fig.tight_layout(rect=(0, 0, 1, 0.965))
            pdf.savefig(fig)
            plt.close(fig)
    return len(ordered)


def plot(result_root: Path) -> None:
    profiles_path = result_root / "report" / "profiles.json"
    records = json.loads(profiles_path.read_text())
    for name, kinds in GROUPS.items():
        output = result_root / "report" / f"absolute_profiles_{name}_all.pdf"
        pages = write_group(records, kinds, output)
        print(f"Wrote {output} ({pages} pages)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_root", type=Path)
    args = parser.parse_args()
    plot(args.result_root)


if __name__ == "__main__":
    main()
