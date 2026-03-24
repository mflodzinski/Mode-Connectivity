from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


def tensor_shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    return None


def summarize_hard_permutation(matrix: torch.Tensor | None) -> dict[str, Any] | None:
    if matrix is None:
        return None
    row_sums = matrix.sum(dim=1)
    col_sums = matrix.sum(dim=0)
    return {
        "shape": list(matrix.shape),
        "row_sum_min": float(row_sums.min().item()),
        "row_sum_max": float(row_sums.max().item()),
        "col_sum_min": float(col_sums.min().item()),
        "col_sum_max": float(col_sums.max().item()),
    }


def to_float_list(value: torch.Tensor | None) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in value.detach().cpu().flatten().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Sinkhorn alignment artifact scale/permutation data.")
    parser.add_argument("artifact_path", type=Path, help="Path to alignment_artifacts.pt")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for analysis outputs. Defaults to <artifact_dir>/scale_analysis",
    )
    args = parser.parse_args()

    artifact_path = args.artifact_path.resolve()
    output_dir = (args.output_dir.resolve() if args.output_dir is not None else artifact_path.parent / "scale_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(artifact_path, map_location="cpu")
    raw_parameters = payload.get("raw_parameters", [])
    hard_permutations = payload.get("hard_permutations", [])
    raw_log_scales = payload.get("raw_log_scales", [])
    scales = payload.get("scales", [])
    inv_scales = payload.get("inv_scales", [])
    layer_scale_stats = payload.get("layer_scale_stats", [])

    summary_rows: list[dict[str, Any]] = []
    scale_value_rows: list[dict[str, Any]] = []

    max_layers = max(
        len(raw_parameters),
        len(hard_permutations),
        len(raw_log_scales),
        len(scales),
        len(inv_scales),
        len(layer_scale_stats),
    )

    for layer_index in range(max_layers):
        layer_stats = layer_scale_stats[layer_index] if layer_index < len(layer_scale_stats) else {}
        raw_parameter = raw_parameters[layer_index] if layer_index < len(raw_parameters) else None
        hard_permutation = hard_permutations[layer_index] if layer_index < len(hard_permutations) else None
        log_scale = raw_log_scales[layer_index] if layer_index < len(raw_log_scales) else None
        scale = scales[layer_index] if layer_index < len(scales) else None
        inv_scale = inv_scales[layer_index] if layer_index < len(inv_scales) else None

        summary_rows.append(
            {
                "layer_index": int(layer_index),
                "parameter_names": layer_stats.get("parameter_names", []),
                "num_channels": int(layer_stats["num_channels"]) if "num_channels" in layer_stats else None,
                "raw_parameter_shape": tensor_shape(raw_parameter),
                "hard_permutation": summarize_hard_permutation(hard_permutation),
                "log_scale_min": layer_stats.get("log_scale_min"),
                "log_scale_max": layer_stats.get("log_scale_max"),
                "log_scale_mean": layer_stats.get("log_scale_mean"),
                "scale_min": layer_stats.get("scale_min"),
                "scale_max": layer_stats.get("scale_max"),
                "scale_mean": layer_stats.get("scale_mean"),
                "inv_scale_min": layer_stats.get("inv_scale_min"),
                "inv_scale_max": layer_stats.get("inv_scale_max"),
                "inv_scale_mean": layer_stats.get("inv_scale_mean"),
            }
        )

        if scale is not None:
            log_scale_values = to_float_list(log_scale)
            scale_values = to_float_list(scale)
            inv_scale_values = to_float_list(inv_scale)
            parameter_names = layer_stats.get("parameter_names", [])
            for channel_index, scale_value in enumerate(scale_values or []):
                scale_value_rows.append(
                    {
                        "layer_index": int(layer_index),
                        "parameter_names": ";".join(parameter_names),
                        "channel_index": int(channel_index),
                        "log_scale": None if log_scale_values is None else float(log_scale_values[channel_index]),
                        "scale": float(scale_value),
                        "inv_scale": None if inv_scale_values is None else float(inv_scale_values[channel_index]),
                    }
                )

    summary_payload = {
        "artifact_path": str(artifact_path),
        "scale_invariant": payload.get("scale_invariant"),
        "lambda_scale": payload.get("lambda_scale"),
        "scale_stats": payload.get("scale_stats"),
        "num_layers_with_permutations": len(raw_parameters),
        "num_layers_with_scales": len(scales),
        "layer_summaries": summary_rows,
        "note": (
            "If num_layers_with_scales is 0, this artifact predates per-layer scale persistence "
            "and only aggregate scale_stats are available."
        ),
    }

    with open(output_dir / "scale_summary.json", "w") as handle:
        json.dump(summary_payload, handle, indent=2)

    with open(output_dir / "layer_scales.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer_index",
                "parameter_names",
                "channel_index",
                "log_scale",
                "scale",
                "inv_scale",
            ],
        )
        writer.writeheader()
        writer.writerows(scale_value_rows)

    with open(output_dir / "layer_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "layer_index",
                "parameter_names",
                "num_channels",
                "raw_parameter_shape",
                "log_scale_min",
                "log_scale_max",
                "log_scale_mean",
                "scale_min",
                "scale_max",
                "scale_mean",
                "inv_scale_min",
                "inv_scale_max",
                "inv_scale_mean",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "layer_index": row["layer_index"],
                    "parameter_names": ";".join(row["parameter_names"]),
                    "num_channels": row["num_channels"],
                    "raw_parameter_shape": row["raw_parameter_shape"],
                    "log_scale_min": row["log_scale_min"],
                    "log_scale_max": row["log_scale_max"],
                    "log_scale_mean": row["log_scale_mean"],
                    "scale_min": row["scale_min"],
                    "scale_max": row["scale_max"],
                    "scale_mean": row["scale_mean"],
                    "inv_scale_min": row["inv_scale_min"],
                    "inv_scale_max": row["inv_scale_max"],
                    "inv_scale_mean": row["inv_scale_mean"],
                }
            )

    print(f"Artifact: {artifact_path}")
    print(f"Summary JSON: {output_dir / 'scale_summary.json'}")
    print(f"Layer summary CSV: {output_dir / 'layer_summary.csv'}")
    print(f"Layer scales CSV: {output_dir / 'layer_scales.csv'}")


if __name__ == "__main__":
    main()
