"""Compare three interpolation variants for one retained VGG/CIFAR endpoint pair.

The runner evaluates naive, permutation-only, and permutation-plus-scale paths
on the same data split and writes the comparison curves used in the thesis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

from mode_connectivity.core import data as core_data

matplotlib.use("Agg")

from mode_connectivity.alignment.permutation_pipeline import resolve_device
from mode_connectivity.core.output import ensure_dir
from mode_connectivity.sinkhorn.shared import (
    evaluate_rebased_curve,
    load_vgg_checkpoint_model,
    plot_three_way_curves,
    save_three_way_curve_arrays,
)
from mode_connectivity.external.sinkhorn_rebasin import import_vgg_rebasin_components


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare naive, perm-only, and perm+scale interpolation curves for CIFAR10 VGG models.")
    parser.add_argument("--vgg-name", type=str, required=True)
    parser.add_argument("--model-a-checkpoint", type=Path, required=True)
    parser.add_argument("--model-b-checkpoint", type=Path, required=True)
    parser.add_argument("--rebased-perm-checkpoint", type=Path, required=True)
    parser.add_argument("--rebased-scale-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-eval-points", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir.resolve())
    device = resolve_device(args.device)
    VGGClass, _, _, _, _, _, _, _, eval_loss_acc, lerp = import_vgg_rebasin_components()
    train_loader, test_loader, eval_transform = core_data.build_cifar10_vgg_eval_loaders(
        data_path=args.data_path.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_a = load_vgg_checkpoint_model(
        args.model_a_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    model_b = load_vgg_checkpoint_model(
        args.model_b_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    rebased_perm = load_vgg_checkpoint_model(
        args.rebased_perm_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    rebased_scale = load_vgg_checkpoint_model(
        args.rebased_scale_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )

    print("Computing naive train curve")
    train_naive = evaluate_rebased_curve(
        model_left=model_a,
        model_right=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing naive test curve")
    test_naive = evaluate_rebased_curve(
        model_left=model_a,
        model_right=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing perm-only train curve")
    train_perm = evaluate_rebased_curve(
        model_left=rebased_perm,
        model_right=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing perm-only test curve")
    test_perm = evaluate_rebased_curve(
        model_left=rebased_perm,
        model_right=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing perm+scale train curve")
    train_scale = evaluate_rebased_curve(
        model_left=rebased_scale,
        model_right=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing perm+scale test curve")
    test_scale = evaluate_rebased_curve(
        model_left=rebased_scale,
        model_right=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )

    curves = {
        "train_naive": train_naive,
        "test_naive": test_naive,
        "train_perm": train_perm,
        "test_perm": test_perm,
        "train_scale": train_scale,
        "test_scale": test_scale,
    }
    save_three_way_curve_arrays(output_dir, curves)

    if not args.skip_plots:
        show_legend = str(args.vgg_name).upper() == "VGG11"
        plot_three_way_curves(
            x=test_naive["lambdas"],
            y_naive=test_naive["losses"],
            y_perm=test_perm["losses"],
            y_scale=test_scale["losses"],
            title=f"{args.vgg_name}: test loss",
            ylabel="Test Loss",
            output_path=output_dir / "compare_test_loss.png",
            show_legend=show_legend,
        )
        plot_three_way_curves(
            x=test_naive["lambdas"],
            y_naive=test_naive["accuracies"],
            y_perm=test_perm["accuracies"],
            y_scale=test_scale["accuracies"],
            title=f"{args.vgg_name}: test accuracy",
            ylabel="Accuracy (%)",
            output_path=output_dir / "compare_test_accuracy.png",
            show_legend=show_legend,
        )
        plot_three_way_curves(
            x=test_naive["lambdas"],
            y_naive=test_naive["errors"],
            y_perm=test_perm["errors"],
            y_scale=test_scale["errors"],
            title=f"{args.vgg_name}: test error",
            ylabel="Error (%)",
            output_path=output_dir / "compare_test_error.png",
            show_legend=show_legend,
        )

    payload = {
        "vgg_name": args.vgg_name,
        "model_a_checkpoint": str(args.model_a_checkpoint.resolve()),
        "model_b_checkpoint": str(args.model_b_checkpoint.resolve()),
        "rebased_perm_checkpoint": str(args.rebased_perm_checkpoint.resolve()),
        "rebased_scale_checkpoint": str(args.rebased_scale_checkpoint.resolve()),
        "data_path": str(args.data_path.resolve()),
        "eval_transform": eval_transform,
        "num_eval_points": int(args.num_eval_points),
        "device": str(device),
        "skip_plots": bool(args.skip_plots),
        "arrays_dir": str((output_dir / "arrays").resolve()),
        "curves": curves,
    }
    with open(output_dir / "curves.json", "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved outputs under: {output_dir}")


if __name__ == "__main__":
    main()
