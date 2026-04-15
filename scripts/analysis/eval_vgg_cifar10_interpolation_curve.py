from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.analysis.compare_vgg_cifar10_three_way_interpolations import (
    build_cifar10_loaders,
    evaluate_curve,
    import_components,
    load_model_from_checkpoint,
)
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one rebased VGG CIFAR10 interpolation curve against endpoint B.")
    parser.add_argument("--vgg-name", type=str, required=True)
    parser.add_argument("--model-b-checkpoint", type=Path, required=True)
    parser.add_argument("--rebased-checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-eval-points", type=int, default=51)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    output_path = args.output_path.resolve()
    ensure_dir(output_path.parent)
    device = resolve_device(args.device)

    VGGClass, dnn_data, eval_loss_acc, lerp = import_components()
    train_loader, test_loader, eval_transform = build_cifar10_loaders(
        data_path=args.data_path.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        dnn_data=dnn_data,
    )

    rebased_model = load_model_from_checkpoint(
        checkpoint_path=args.rebased_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    model_b = load_model_from_checkpoint(
        checkpoint_path=args.model_b_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )

    print("Computing train curve")
    train_curve = evaluate_curve(
        model_a=rebased_model,
        model_b=model_b,
        loader=train_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )
    print("Computing test curve")
    test_curve = evaluate_curve(
        model_a=rebased_model,
        model_b=model_b,
        loader=test_loader,
        num_eval_points=args.num_eval_points,
        eval_loss_acc=eval_loss_acc,
        lerp=lerp,
        device=device,
    )

    payload = {
        "vgg_name": args.vgg_name,
        "model_b_checkpoint": str(args.model_b_checkpoint.resolve()),
        "rebased_checkpoint": str(args.rebased_checkpoint.resolve()),
        "data_path": str(args.data_path.resolve()),
        "eval_transform": eval_transform,
        "num_eval_points": int(args.num_eval_points),
        "device": str(device),
        "train_curve": train_curve,
        "test_curve": test_curve,
    }
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2)
    print(f"Saved interpolation curves to: {output_path}")


if __name__ == "__main__":
    main()
