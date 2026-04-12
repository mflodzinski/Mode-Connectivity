from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.experiments.xor_curve_fitting import (
    compute_linear_path,
    compute_path_vectors_linear,
    save_curve_npz,
    summarize_barriers,
    train_xor_network,
)
from scripts.experiments.xor_permutation_scale_experiment import clone_state_cpu, ensure_dir, parse_pairs, parse_seeds


def plot_pair_bars(pair_results: list[dict[str, Any]], output_path: Path, key: str, ylabel: str, title: str) -> None:
    labels = [f"{pair['seed_a']}-{pair['seed_b']}" for pair in pair_results]
    values = [float(pair["linear_metrics"][key]) for pair in pair_results]
    plt.figure(figsize=(max(8, 0.6 * len(labels)), 4.8))
    plt.bar(range(len(labels)), values, color="tab:blue")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XOR models, keep only perfect ones, and evaluate linear interpolation barriers for all kept pairs")
    parser.add_argument("--output", type=Path, default=Path("results/xor/xor_5h_trained_linear_pairs"))
    parser.add_argument("--hidden-size", type=int, default=5)
    parser.add_argument("--num-networks", type=int, default=10)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--pairs", type=str, default=None)
    parser.add_argument("--max-endpoint-loss", type=float, default=0.02)
    parser.add_argument("--train-max-epochs", type=int, default=None)
    parser.add_argument("--train-lr", type=float, default=None)
    parser.add_argument("--curve-eval-points", type=int, default=61)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output.resolve())
    checkpoints_dir = ensure_dir(output_dir / "checkpoints")
    evaluations_dir = ensure_dir(output_dir / "evaluations")
    plots_dir = ensure_dir(output_dir / "plots")

    requested_seeds = parse_seeds(args.seeds, args.num_networks)
    print("=" * 80)
    print("XOR TRAIN + LINEAR PAIR BARRIERS")
    print("=" * 80)
    print(f"hidden_size: {args.hidden_size}")
    print(f"requested_seeds: {requested_seeds}")
    print(f"max_endpoint_loss: {args.max_endpoint_loss}")
    print("")

    models_by_seed: dict[int, torch.nn.Module] = {}
    endpoint_results: dict[int, dict[str, float]] = {}
    rejected: dict[int, dict[str, float]] = {}

    print(f"Step 1/3: training {len(requested_seeds)} XOR models")
    for seed in requested_seeds:
        model, eval_res = train_xor_network(
            seed=seed,
            hidden_size=args.hidden_size,
            max_epochs=args.train_max_epochs,
            lr=args.train_lr,
            verbose=args.verbose,
        )
        if eval_res["accuracy"] < 100.0 or eval_res["loss"] > float(args.max_endpoint_loss):
            rejected[seed] = eval_res
            if args.verbose:
                print(f"  seed {seed}: rejected (acc={eval_res['accuracy']:.1f} loss={eval_res['loss']:.6f})")
            continue

        models_by_seed[seed] = model
        endpoint_results[seed] = eval_res
        torch.save(
            {
                "seed": seed,
                "hidden_size": args.hidden_size,
                "state_dict": clone_state_cpu(model.state_dict()),
                "eval": eval_res,
            },
            checkpoints_dir / f"seed_{seed}.pt",
        )

    available_seeds = sorted(models_by_seed)
    if len(available_seeds) < 2:
        raise RuntimeError(f"Need at least 2 kept XOR models, got {len(available_seeds)}.")

    pairs = parse_pairs(args.pairs, available_seeds)
    print(f"Step 2/3: computing linear interpolation for {len(pairs)} pairs from kept seeds {available_seeds}")

    pair_results: list[dict[str, Any]] = []
    for pair_index, (seed_a, seed_b) in enumerate(pairs, start=1):
        print(f"[pair {pair_index:02d}/{len(pairs):02d}] {seed_a}-{seed_b}")
        model_a = models_by_seed[seed_a]
        model_b = models_by_seed[seed_b]
        linear_metrics = compute_linear_path(model_a, model_b, num_points=args.curve_eval_points)

        pair_dir = ensure_dir(evaluations_dir / f"pair_{seed_a}_{seed_b}" / "linear")
        path_vectors = compute_path_vectors_linear(model_a.state_dict(), model_b.state_dict(), linear_metrics["t"])
        npz_path = pair_dir / "curve.npz"
        save_curve_npz(str(npz_path), linear_metrics["t"], linear_metrics["loss"], linear_metrics["accuracy"], path_vectors)

        pair_payload = {
            "seed_a": seed_a,
            "seed_b": seed_b,
            "linear_metrics": linear_metrics,
            "npz_path": str(npz_path),
        }
        pair_results.append(pair_payload)
        with open(pair_dir.parent / "pair_results.json", "w") as handle:
            json.dump(pair_payload, handle, indent=2)

    print("Step 3/3: writing summaries and plots")
    summary = {
        "num_requested_seeds": len(requested_seeds),
        "num_kept_seeds": len(available_seeds),
        "kept_seeds": available_seeds,
        "num_pairs": len(pair_results),
        "loss_barriers": summarize_barriers([pair["linear_metrics"]["loss_barrier"] for pair in pair_results]),
        "accuracy_barriers": summarize_barriers([pair["linear_metrics"]["barrier"] for pair in pair_results]),
        "mean_interpolation_loss": summarize_barriers([float(np.mean(pair["linear_metrics"]["loss"])) for pair in pair_results]),
        "min_interpolation_accuracy": summarize_barriers([pair["linear_metrics"]["min_accuracy"] for pair in pair_results]),
    }

    plot_pair_bars(
        pair_results,
        plots_dir / "pair_loss_barriers.png",
        key="loss_barrier",
        ylabel="Loss barrier",
        title="XOR Linear Interpolation Loss Barrier per Pair",
    )
    plot_pair_bars(
        pair_results,
        plots_dir / "pair_accuracy_barriers.png",
        key="barrier",
        ylabel="Accuracy barrier (%)",
        title="XOR Linear Interpolation Accuracy Barrier per Pair",
    )

    results = {
        "config": {
            "hidden_size": int(args.hidden_size),
            "requested_seeds": requested_seeds,
            "pairs": [list(pair) for pair in pairs],
            "max_endpoint_loss": float(args.max_endpoint_loss),
            "curve_eval_points": int(args.curve_eval_points),
            "train_max_epochs": args.train_max_epochs,
            "train_lr": args.train_lr,
        },
        "endpoint_results": endpoint_results,
        "rejected_results": rejected,
        "pair_results": pair_results,
        "summary": summary,
        "artifacts": {
            "checkpoints_dir": str(checkpoints_dir),
            "evaluations_dir": str(evaluations_dir),
            "plots_dir": str(plots_dir),
            "pair_loss_barriers_plot": str(plots_dir / "pair_loss_barriers.png"),
            "pair_accuracy_barriers_plot": str(plots_dir / "pair_accuracy_barriers.png"),
        },
    }
    with open(output_dir / "xor_train_linear_results.json", "w") as handle:
        json.dump(results, handle, indent=2)

    print("")
    print("=" * 80)
    print("XOR TRAIN + LINEAR PAIR SUMMARY")
    print("=" * 80)
    print(f"Kept seeds: {available_seeds}")
    print(f"Pairs evaluated: {len(pair_results)}")
    print(f"Mean linear loss barrier: {summary['loss_barriers']['mean']:.6f} +/- {summary['loss_barriers']['std']:.6f}")
    print(f"Results written under: {output_dir}")


if __name__ == "__main__":
    main()
