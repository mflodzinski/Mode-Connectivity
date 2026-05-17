from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mode_connectivity.alignment.permutation_pipeline import resolve_device
from mode_connectivity.core import data as core_data
from mode_connectivity.core.output import ensure_dir
from mode_connectivity.external.sinkhorn_rebasin import import_vgg_rebasin_components
from mode_connectivity.sinkhorn.shared import build_vgg_model, load_model_from_checkpoint as load_model_with_factory


def load_model_from_checkpoint(
    *,
    checkpoint_path: Path,
    vgg_name: str,
    num_classes: int,
    image_size: int,
    device: torch.device,
    VGGClass,
) -> torch.nn.Module:
    return load_model_with_factory(
        checkpoint_path,
        model_factory=lambda: build_vgg_model(VGGClass, vgg_name, num_classes=num_classes, image_size=image_size),
        device=device,
    )


def evaluate_equivalence(
    *,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    total_examples = 0
    num_logits = None
    total_loss_a = 0.0
    total_loss_b = 0.0
    total_correct_a = 0
    total_correct_b = 0
    total_prediction_agreement = 0
    total_mean_abs_logit_diff = 0.0
    total_mean_sq_logit_diff = 0.0
    global_max_abs_logit_diff = 0.0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits_a = model_a(x)
            logits_b = model_b(x)
            if num_logits is None:
                num_logits = x.shape[0] * logits_a.shape[1]
            else:
                num_logits += x.shape[0] * logits_a.shape[1]

            total_loss_a += float(criterion(logits_a, y).item())
            total_loss_b += float(criterion(logits_b, y).item())
            total_correct_a += int(logits_a.argmax(dim=1).eq(y).sum().item())
            total_correct_b += int(logits_b.argmax(dim=1).eq(y).sum().item())
            total_prediction_agreement += int(logits_a.argmax(dim=1).eq(logits_b.argmax(dim=1)).sum().item())

            abs_diff = (logits_a - logits_b).abs()
            total_mean_abs_logit_diff += float(abs_diff.sum().item())
            total_mean_sq_logit_diff += float(torch.pow(logits_a - logits_b, 2).sum().item())
            global_max_abs_logit_diff = max(global_max_abs_logit_diff, float(abs_diff.max().item()))
            total_examples += x.shape[0]

    return {
        "test_loss_model_a": total_loss_a / total_examples,
        "test_loss_rebased": total_loss_b / total_examples,
        "test_acc_model_a": 100.0 * total_correct_a / total_examples,
        "test_acc_rebased": 100.0 * total_correct_b / total_examples,
        "prediction_agreement": 100.0 * total_prediction_agreement / total_examples,
        "mean_abs_logit_diff": total_mean_abs_logit_diff / num_logits,
        "rmse_logit_diff": (total_mean_sq_logit_diff / num_logits) ** 0.5,
        "max_abs_logit_diff": global_max_abs_logit_diff,
        "num_examples": total_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check functional equivalence between two VGG/CIFAR10 checkpoints."
    )
    parser.add_argument("--model-a-checkpoint", type=Path, required=True)
    parser.add_argument("--rebased-checkpoint", type=Path, required=True)
    parser.add_argument("--vgg-name", type=str, default="VGG11")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    VGGClass, _, _, _, _, _, _, _, _, _ = import_vgg_rebasin_components()
    _, test_loader, _ = core_data.build_cifar10_vgg_eval_loaders(
        data_path=args.data_path.resolve(),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model_a = load_model_from_checkpoint(
        checkpoint_path=args.model_a_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    rebased_model = load_model_from_checkpoint(
        checkpoint_path=args.rebased_checkpoint.resolve(),
        vgg_name=args.vgg_name,
        num_classes=args.num_classes,
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )

    results = {
        "model_a_checkpoint": str(args.model_a_checkpoint.resolve()),
        "rebased_checkpoint": str(args.rebased_checkpoint.resolve()),
        "vgg_name": args.vgg_name,
        "num_classes": args.num_classes,
        "device": str(device),
        **evaluate_equivalence(model_a=model_a, model_b=rebased_model, loader=test_loader, device=device),
    }

    print(json.dumps(results, indent=2))

    if args.output_json is not None:
        output_json = args.output_json.resolve()
        ensure_dir(output_json.parent)
        with open(output_json, "w") as handle:
            json.dump(results, handle, indent=2)
        print(f"Saved results to: {output_json}")


if __name__ == "__main__":
    main()
