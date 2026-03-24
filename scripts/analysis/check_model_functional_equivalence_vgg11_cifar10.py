from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torchvision

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import build_model, import_original_mnist_components
from scripts.lib.alignment.permutation_pipeline import resolve_device
from scripts.lib.core.output import ensure_dir


def load_model_from_checkpoint(*, checkpoint_path: Path, image_size: int, device: torch.device, VGGClass) -> torch.nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError(
            f"Unsupported checkpoint payload at {checkpoint_path}; expected raw state_dict or dict with "
            "'model_state'/'state_dict'."
        )

    model = build_model(VGGClass, "VGG11", num_classes=10, image_size=image_size)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def evaluate_equivalence(
    *,
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: torch.device,
) -> dict[str, float]:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    total_examples = 0
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

    num_logits = total_examples * 10
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
    parser = argparse.ArgumentParser(description="Check functional equivalence between VGG11/CIFAR10 endpoint and rebased model.")
    parser.add_argument("--model-a-checkpoint", type=Path, required=True)
    parser.add_argument("--rebased-checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("./data"))
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    VGGClass, _, _, dnn_data, _, _, _ = import_original_mnist_components()
    transform_test = dnn_data.Transforms.CIFAR10.VGG.test
    cifar_root = args.data_path.resolve() / "cifar10"
    dataset_test = torchvision.datasets.CIFAR10(
        root=cifar_root,
        train=False,
        download=True,
        transform=transform_test,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_a = load_model_from_checkpoint(
        checkpoint_path=args.model_a_checkpoint.resolve(),
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )
    rebased_model = load_model_from_checkpoint(
        checkpoint_path=args.rebased_checkpoint.resolve(),
        image_size=args.image_size,
        device=device,
        VGGClass=VGGClass,
    )

    results = {
        "model_a_checkpoint": str(args.model_a_checkpoint.resolve()),
        "rebased_checkpoint": str(args.rebased_checkpoint.resolve()),
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
