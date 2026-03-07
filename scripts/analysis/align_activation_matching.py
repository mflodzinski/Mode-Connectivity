"""Permutation alignment for independently trained VGG16 endpoints.

Supports:
- activation matching (git-rebasin style, via activation correlations)
- weight matching (git-rebasin iterative algorithm on weights)

Then it evaluates linear interpolation barriers (LMC proxy) before/after
alignment.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# Add project root to import scripts.lib.*
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
import sys
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from scripts.lib.alignment.permutation_spec import vgg16_permutation_spec
from scripts.lib.alignment.weight_matching import apply_permutation, weight_matching
from scripts.lib.analysis.alignment import (
    compute_state_dict_l2_distance,
    create_vgg16_model,
    load_cifar10_eval_loaders,
    load_vgg16_model,
    state_dict_to_perm_params,
)


def vgg16_activation_module_map() -> Dict[str, str]:
    """Map permutation keys to post-ReLU module names in PyTorch VGG16."""
    mapping: Dict[str, str] = {}

    convs_per_block = [2, 2, 3, 3, 3]
    conv_idx = 0
    for block_idx, num_layers in enumerate(convs_per_block):
        for layer_idx in range(num_layers):
            mapping[f"P_Conv_{conv_idx}"] = f"activation_blocks.{block_idx}.{layer_idx}"
            conv_idx += 1

    mapping["P_Dense_0"] = "classifier.2"
    mapping["P_Dense_1"] = "classifier.5"
    return mapping


class ActivationTap:
    """Capture forward outputs from selected modules."""

    def __init__(self, model: torch.nn.Module, module_name_map: Dict[str, str]):
        self.model = model
        self.module_name_map = module_name_map
        self.outputs: Dict[str, torch.Tensor] = {}
        self.handles = []

        modules = dict(model.named_modules())
        for layer_key, module_name in module_name_map.items():
            if module_name not in modules:
                raise KeyError(f"Module '{module_name}' not found in model")
            module = modules[module_name]
            handle = module.register_forward_hook(self._hook(layer_key))
            self.handles.append(handle)

    def _hook(self, layer_key: str):
        def _capture(_module, _inputs, output):
            self.outputs[layer_key] = output.detach()
        return _capture

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


@dataclass
class OnlineMoments:
    """Streaming stats for channel-wise cross-model correlation."""

    sum_a: torch.Tensor
    sum_b: torch.Tensor
    sumsq_a: torch.Tensor
    sumsq_b: torch.Tensor
    sum_ab: torch.Tensor
    count: int

    @staticmethod
    def init(num_channels: int) -> "OnlineMoments":
        zeros = torch.zeros(num_channels, dtype=torch.float64)
        return OnlineMoments(
            sum_a=zeros.clone(),
            sum_b=zeros.clone(),
            sumsq_a=zeros.clone(),
            sumsq_b=zeros.clone(),
            sum_ab=torch.zeros((num_channels, num_channels), dtype=torch.float64),
            count=0,
        )

    def update(self, a: torch.Tensor, b: torch.Tensor) -> None:
        # a,b: [N, C] on CPU float64
        self.sum_a += a.sum(dim=0)
        self.sum_b += b.sum(dim=0)
        self.sumsq_a += (a * a).sum(dim=0)
        self.sumsq_b += (b * b).sum(dim=0)
        self.sum_ab += a.T @ b
        self.count += a.shape[0]

    def pearson_corr(self) -> torch.Tensor:
        if self.count == 0:
            raise RuntimeError("No activation samples were collected")

        n = float(self.count)
        mean_a = self.sum_a / n
        mean_b = self.sum_b / n

        # Numerator/variance in unnormalized form; n/(n-1) cancels in Pearson.
        cov_num = self.sum_ab - n * torch.outer(mean_a, mean_b)
        var_a_num = self.sumsq_a - n * (mean_a * mean_a)
        var_b_num = self.sumsq_b - n * (mean_b * mean_b)

        eps = 1e-12
        denom = torch.sqrt(torch.clamp(var_a_num, min=eps)).unsqueeze(1)
        denom = denom * torch.sqrt(torch.clamp(var_b_num, min=eps)).unsqueeze(0)
        corr = cov_num / denom
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return corr


def flatten_activations(x: torch.Tensor) -> torch.Tensor:
    """Convert activations to [N, C] format."""
    if x.ndim == 2:
        # [B, C]
        return x
    if x.ndim == 4:
        # [B, C, H, W] -> [B*H*W, C]
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    raise ValueError(f"Unsupported activation shape: {tuple(x.shape)}")


def subsample_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    max_rows: int,
    rng: np.random.RandomState,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample rows consistently across a/b if needed."""
    n = a.shape[0]
    if max_rows <= 0 or n <= max_rows:
        return a, b

    idx_np = rng.choice(n, size=max_rows, replace=False)
    idx = torch.from_numpy(idx_np).to(a.device)
    return a.index_select(0, idx), b.index_select(0, idx)


def compute_activation_permutation(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
    max_rows_per_batch: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Compute per-layer permutation using activation correlations."""
    layer_map = vgg16_activation_module_map()
    tap_a = ActivationTap(model_a, layer_map)
    tap_b = ActivationTap(model_b, layer_map)
    moments: Dict[str, OnlineMoments] = {}
    rng = np.random.RandomState(seed)

    model_a.eval()
    model_b.eval()

    try:
        with torch.no_grad():
            for batch_idx, (inputs, _targets) in enumerate(loader):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                inputs = inputs.to(device, non_blocking=True)

                tap_a.outputs.clear()
                tap_b.outputs.clear()
                _ = model_a(inputs)
                _ = model_b(inputs)

                for layer_key in layer_map:
                    a = flatten_activations(tap_a.outputs[layer_key])
                    b = flatten_activations(tap_b.outputs[layer_key])
                    a, b = subsample_rows(a, b, max_rows_per_batch, rng)

                    a = a.to("cpu", dtype=torch.float64)
                    b = b.to("cpu", dtype=torch.float64)

                    if layer_key not in moments:
                        moments[layer_key] = OnlineMoments.init(a.shape[1])
                    moments[layer_key].update(a, b)
    finally:
        tap_a.close()
        tap_b.close()

    permutation: Dict[str, np.ndarray] = {}
    print("\nActivation matching summary (Pearson correlation):")
    for layer_key in layer_map:
        corr = moments[layer_key].pearson_corr().numpy()
        ri, ci = linear_sum_assignment(corr, maximize=True)
        if not np.all(ri == np.arange(len(ri))):
            raise RuntimeError(f"Unexpected row assignment for layer {layer_key}")

        diag_before = float(np.mean(np.diag(corr)))
        diag_after = float(np.mean(corr[np.arange(corr.shape[0]), ci]))
        print(
            f"  {layer_key:>10s}: mean diag before={diag_before:+.4f}, "
            f"after={diag_after:+.4f}, delta={diag_after - diag_before:+.4f}"
        )
        permutation[layer_key] = ci.astype(np.int64)

    return permutation


def compute_weight_permutation(
    state_a: OrderedDict[str, torch.Tensor],
    state_b: OrderedDict[str, torch.Tensor],
    max_iter: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    """Compute permutation with iterative weight matching."""
    ps = vgg16_permutation_spec()
    params_a = state_dict_to_perm_params(state_a, ps)
    params_b = state_dict_to_perm_params(state_b, ps)
    return weight_matching(
        ps=ps,
        params_a=params_a,
        params_b=params_b,
        max_iter=max_iter,
        seed=seed,
        silent=False,
    )


def apply_vgg16_permutation_to_state(
    state_dict: OrderedDict[str, torch.Tensor],
    permutation: Dict[str, np.ndarray],
) -> OrderedDict[str, torch.Tensor]:
    """Apply permutation spec-consistently and return a new state dict."""
    ps = vgg16_permutation_spec()
    params = state_dict_to_perm_params(state_dict, ps)
    params_aligned = apply_permutation(ps, permutation, params)

    aligned_state = OrderedDict((k, v.clone()) for k, v in state_dict.items())
    for key, value in params_aligned.items():
        aligned_state[key] = value
    return aligned_state


def evaluate_model_limited(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    """Evaluate model on a loader with optional batch cap."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
            total_loss += float(loss.item())
            total_correct += int(logits.argmax(dim=1).eq(targets).sum().item())
            total_examples += int(targets.shape[0])

    if total_examples == 0:
        raise RuntimeError("No evaluation samples were processed")

    return {
        "loss": total_loss / total_examples,
        "accuracy": 100.0 * total_correct / total_examples,
    }


def evaluate_test_barrier_limited(
    model_a: torch.nn.Module,
    model_b: torch.nn.Module,
    loader,
    device: torch.device,
    num_points: int,
    max_batches: int,
) -> Dict[str, object]:
    """Compute linear interpolation barrier on a (possibly capped) eval loader."""
    model_a = model_a.to(device)
    model_b = model_b.to(device)
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    interp_model = create_vgg16_model(num_classes=10, device=device)

    ts = np.linspace(0.0, 1.0, num_points)
    test_loss = []
    test_acc = []

    for t in ts:
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1.0 - t) * state_a[key] + t * state_b[key]
        interp_model.load_state_dict(interp_state)

        metrics = evaluate_model_limited(interp_model, loader, device, max_batches=max_batches)
        test_loss.append(float(metrics["loss"]))
        test_acc.append(float(metrics["accuracy"]))

    endpoint_avg = 0.5 * (test_loss[0] + test_loss[-1])
    max_loss = float(max(test_loss))
    min_acc = float(min(test_acc))

    return {
        "t": ts.tolist(),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "barrier": max_loss - endpoint_avg,
        "max_test_loss": max_loss,
        "endpoint_avg_test_loss": endpoint_avg,
        "min_test_acc": min_acc,
    }


def main():
    parser = argparse.ArgumentParser(description="Permutation alignment for VGG16 CIFAR-10 checkpoints")
    parser.add_argument("--model-a", type=str, required=True, help="Reference checkpoint path")
    parser.add_argument("--model-b", type=str, required=True, help="Checkpoint path to align")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store outputs")
    parser.add_argument(
        "--method",
        type=str,
        default="activation_matching",
        choices=["activation_matching", "weight_matching"],
        help="Alignment method",
    )
    parser.add_argument("--data-path", type=str, default="./data", help="Dataset path")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for stats/eval")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument("--max-batches", type=int, default=100, help="Activation batches; <=0 uses full train set")
    parser.add_argument("--max-rows-per-batch", type=int, default=8192, help="Activation rows per layer per batch")
    parser.add_argument("--wm-max-iter", type=int, default=100, help="Weight matching max iterations")
    parser.add_argument("--num-eval-points", type=int, default=11, help="Interpolation points for LMC check")
    parser.add_argument("--eval-max-batches", type=int, default=20, help="Max test batches per interpolation point")
    parser.add_argument("--lmc-threshold", type=float, default=0.1, help="Barrier threshold for LMC")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for activation row subsampling")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading checkpoints:\n  A={args.model_a}\n  B={args.model_b}")

    model_a = load_vgg16_model(args.model_a).to(device)
    model_b = load_vgg16_model(args.model_b).to(device)

    loaders, _num_classes = load_cifar10_eval_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    if args.method == "activation_matching":
        print("\nComputing activation-matching permutation...")
        perm = compute_activation_permutation(
            model_a=model_a,
            model_b=model_b,
            loader=loaders["train"],
            device=device,
            max_batches=args.max_batches,
            max_rows_per_batch=args.max_rows_per_batch,
            seed=args.seed,
        )
    else:
        print("\nComputing weight-matching permutation...")
        perm = compute_weight_permutation(
            state_a=state_a,
            state_b=state_b,
            max_iter=args.wm_max_iter,
            seed=args.seed,
        )

    print("\nApplying permutation to model B...")
    state_b_aligned = apply_vgg16_permutation_to_state(state_b, perm)

    model_b_aligned = load_vgg16_model(args.model_b).to(device)
    model_b_aligned.load_state_dict(state_b_aligned)

    # Save aligned checkpoint (preserve original checkpoint fields where possible).
    ckpt_b = torch.load(args.model_b, map_location="cpu")
    ckpt_b["model_state"] = OrderedDict((k, v.cpu()) for k, v in state_b_aligned.items())
    method_tag = "activation-matched" if args.method == "activation_matching" else "weight-matched"
    aligned_ckpt_path = os.path.join(args.output_dir, f"checkpoint-200-{method_tag}.pt")
    torch.save(ckpt_b, aligned_ckpt_path)
    print(f"Aligned checkpoint saved to: {aligned_ckpt_path}")

    # Save permutation
    perm_npz_path = os.path.join(args.output_dir, f"{args.method}_permutation.npz")
    np.savez(perm_npz_path, **perm)
    print(f"Permutation saved to: {perm_npz_path}")

    print("\nEvaluating linear barriers (LMC proxy, test split)...")
    barrier_before = evaluate_test_barrier_limited(
        model_a=model_a,
        model_b=model_b,
        loader=loaders["test"],
        device=device,
        num_points=args.num_eval_points,
        max_batches=args.eval_max_batches,
    )
    barrier_after = evaluate_test_barrier_limited(
        model_a=model_a,
        model_b=model_b_aligned,
        loader=loaders["test"],
        device=device,
        num_points=args.num_eval_points,
        max_batches=args.eval_max_batches,
    )

    dist_before = compute_state_dict_l2_distance(state_a, state_b)
    dist_after = compute_state_dict_l2_distance(state_a, state_b_aligned)

    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "aligned_model_b": aligned_ckpt_path,
        "method": args.method,
        "matching": {
            "seed": args.seed,
            "activation_matching": {
                "max_batches": args.max_batches,
                "max_rows_per_batch": args.max_rows_per_batch,
            },
            "weight_matching": {
                "max_iter": args.wm_max_iter,
            },
        },
        "lmc_eval": {
            "split": "test",
            "num_eval_points": args.num_eval_points,
            "max_eval_batches": args.eval_max_batches,
        },
        "lmc_threshold": args.lmc_threshold,
        "before_alignment": {
            "distance": dist_before,
            "barrier": barrier_before,
            "is_lmc": barrier_before["barrier"] < args.lmc_threshold,
        },
        "after_alignment": {
            "distance": dist_after,
            "barrier": barrier_after,
            "is_lmc": barrier_after["barrier"] < args.lmc_threshold,
        },
        "improvement": {
            "barrier_delta": barrier_before["barrier"] - barrier_after["barrier"],
            "barrier_relative_reduction_percent": (
                100.0 * (barrier_before["barrier"] - barrier_after["barrier"]) /
                barrier_before["barrier"]
            ) if abs(barrier_before["barrier"]) > 1e-12 else 0.0,
            "l2_delta": dist_before["l2_distance"] - dist_after["l2_distance"],
        },
    }

    summary_path = os.path.join(args.output_dir, f"{args.method}_lmc_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print("\n=== Final metrics ===")
    print(
        "Before alignment: "
        f"barrier={barrier_before['barrier']:.6f}, "
        f"min_test_acc={barrier_before['min_test_acc']:.2f}%"
    )
    print(
        "After alignment:  "
        f"barrier={barrier_after['barrier']:.6f}, "
        f"min_test_acc={barrier_after['min_test_acc']:.2f}%"
    )
    print(
        f"LMC status (threshold={args.lmc_threshold:.4f}): "
        f"before={summary['before_alignment']['is_lmc']}, "
        f"after={summary['after_alignment']['is_lmc']}"
    )


if __name__ == "__main__":
    main()
