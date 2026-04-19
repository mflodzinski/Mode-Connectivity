"""Permutation alignment for independently trained VGG endpoints.

Supports:
- activation matching (git-rebasin style, via activation correlations)
- weight matching (git-rebasin iterative algorithm on weights)

The script supports two checkpoint layouts:
- legacy ``layer_blocks.*`` VGG checkpoints from this repo
- external ``features.*`` VGG checkpoints from ``external/pytorch-vgg-cifar10``

It saves:
- aligned checkpoint
- permutation artifact
- functional equivalence report
- train/test interpolation curves before and after alignment
- before/after interpolation plots
- summary JSON with barrier metrics
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import matplotlib
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

project_root = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(project_root / ".mplcache"))
os.environ.setdefault("XDG_CACHE_HOME", str(project_root / ".mplcache"))

import sys

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.lib.alignment.permutation_spec import (
    PermutationSpec,
    permutation_spec_from_axes_to_perm,
    vgg16_features_permutation_spec,
    vgg16_permutation_spec,
)
from scripts.lib.alignment.weight_matching import apply_permutation, weight_matching
from scripts.lib.analysis.alignment import (
    compute_state_dict_l2_distance,
    create_vgg16_model,
    load_cifar10_eval_loaders,
    load_vgg16_model,
    state_dict_to_perm_params,
)


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized_state_dict = {}
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("module."):
            normalized_key = normalized_key[len("module.") :]
        if normalized_key.startswith("features.module."):
            normalized_key = "features." + normalized_key[len("features.module.") :]
        normalized_state_dict[normalized_key] = value
    return normalized_state_dict


def extract_state_dict(payload: object) -> OrderedDict[str, torch.Tensor]:
    if isinstance(payload, dict) and "model_state" in payload:
        state_dict = payload["model_state"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise ValueError("Unsupported checkpoint payload; expected raw state_dict or dict with model_state/state_dict.")
    return OrderedDict((key, value) for key, value in normalize_state_dict_keys(state_dict).items())


ARCH_BLOCK_CONV_COUNTS = {
    "VGG11": [1, 1, 2, 2, 2],
    "VGG13": [2, 2, 2, 2, 2],
    "VGG16": [2, 2, 3, 3, 3],
    "VGG19": [2, 2, 4, 4, 4],
}

CONV_COUNT_TO_VGG_NAME = {
    sum(block_counts): vgg_name for vgg_name, block_counts in ARCH_BLOCK_CONV_COUNTS.items()
}


def infer_layout(state_dict: Dict[str, torch.Tensor]) -> str:
    keys = set(state_dict.keys())
    if any(key.startswith("layer_blocks.") for key in keys):
        return "layer_blocks"
    if any(key.startswith("features.") for key in keys):
        return "features"
    raise ValueError(
        "Unable to infer VGG checkpoint layout. Expected keys starting with 'layer_blocks.' or 'features.'."
    )


def import_external_vgg_class():
    sinkhorn_root = project_root / "external" / "sinkhorn-rebasin"
    examples_root = sinkhorn_root / "examples"
    dnn_root = project_root / "external" / "dnn-mode-connectivity"
    for path in (str(examples_root), str(sinkhorn_root), str(dnn_root)):
        if path not in sys.path:
            sys.path.insert(0, path)

    sinkhorn_vgg_path = examples_root / "models" / "vgg.py"
    spec = importlib.util.spec_from_file_location("_sinkhorn_rebasin_examples_vgg_for_alignment", sinkhorn_vgg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load sinkhorn VGG definition from {sinkhorn_vgg_path}.")
    vgg_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vgg_module)
    return vgg_module.VGG


def build_external_vgg_model(vgg_name: str, device: torch.device | None = None) -> torch.nn.Module:
    VGGClass = import_external_vgg_class()
    model = VGGClass(vgg_name, in_channels=3, out_features=10, h_in=32, w_in=32)
    if device is not None:
        model = model.to(device)
    return model


def infer_external_vgg_name(state_dict: Dict[str, torch.Tensor]) -> str:
    conv_weight_count = 0
    for key, value in state_dict.items():
        if key.startswith("features.") and key.endswith(".weight") and value.ndim == 4:
            conv_weight_count += 1
    if conv_weight_count not in CONV_COUNT_TO_VGG_NAME:
        raise ValueError(
            f"Unsupported external VGG conv count {conv_weight_count}. "
            f"Expected one of {sorted(CONV_COUNT_TO_VGG_NAME)}."
        )
    return CONV_COUNT_TO_VGG_NAME[conv_weight_count]


def infer_layer_blocks_vgg_name(state_dict: Dict[str, torch.Tensor]) -> str:
    max_block_idx = -1
    conv_count = 0
    for key, value in state_dict.items():
        if key.startswith("layer_blocks.") and key.endswith(".weight") and value.ndim == 4:
            parts = key.split(".")
            max_block_idx = max(max_block_idx, int(parts[1]))
            conv_count += 1
    if max_block_idx != 4 or conv_count not in CONV_COUNT_TO_VGG_NAME:
        raise ValueError(
            f"Unsupported layer_blocks checkpoint with {conv_count} conv layers and max block idx {max_block_idx}."
        )
    return CONV_COUNT_TO_VGG_NAME[conv_count]


def external_conv_relu_indices(block_conv_counts: list[int]) -> tuple[list[int], list[int]]:
    conv_indices: list[int] = []
    relu_indices: list[int] = []
    current_idx = 0
    for block_size in block_conv_counts:
        for _ in range(block_size):
            conv_indices.append(current_idx)
            relu_indices.append(current_idx + 1)
            current_idx += 2
        current_idx += 1  # MaxPool
    return conv_indices, relu_indices


def build_features_permutation_spec(block_conv_counts: list[int]) -> PermutationSpec:
    conv_indices, _ = external_conv_relu_indices(block_conv_counts)
    axes_to_perm = {}

    first_conv = conv_indices[0]
    axes_to_perm[f"features.{first_conv}.weight"] = ("P_Conv_0", None, None, None)
    axes_to_perm[f"features.{first_conv}.bias"] = ("P_Conv_0",)

    for i in range(1, len(conv_indices)):
        curr = conv_indices[i]
        axes_to_perm[f"features.{curr}.weight"] = (
            f"P_Conv_{i}",
            f"P_Conv_{i-1}",
            None,
            None,
        )
        axes_to_perm[f"features.{curr}.bias"] = (f"P_Conv_{i}",)

    axes_to_perm["classifier.1.weight"] = ("P_Dense_0", "P_Conv_" + str(len(conv_indices) - 1))
    axes_to_perm["classifier.1.bias"] = ("P_Dense_0",)
    axes_to_perm["classifier.4.weight"] = ("P_Dense_1", "P_Dense_0")
    axes_to_perm["classifier.4.bias"] = ("P_Dense_1",)
    axes_to_perm["classifier.6.weight"] = (None, "P_Dense_1")
    axes_to_perm["classifier.6.bias"] = (None,)
    return permutation_spec_from_axes_to_perm(axes_to_perm)


def build_layer_blocks_permutation_spec(block_conv_counts: list[int]) -> PermutationSpec:
    mapping: Dict[str, str] = {}
    conv_idx = 0
    axes_to_perm = {}

    for block_idx, num_layers in enumerate(block_conv_counts):
        for layer_idx in range(num_layers):
            weight_key = f"layer_blocks.{block_idx}.{layer_idx}.weight"
            bias_key = f"layer_blocks.{block_idx}.{layer_idx}.bias"
            if conv_idx == 0:
                axes_to_perm[weight_key] = ("P_Conv_0", None, None, None)
            else:
                axes_to_perm[weight_key] = (f"P_Conv_{conv_idx}", f"P_Conv_{conv_idx-1}", None, None)
            axes_to_perm[bias_key] = (f"P_Conv_{conv_idx}",)
            conv_idx += 1

    axes_to_perm["classifier.1.weight"] = ("P_Dense_0", "P_Conv_" + str(conv_idx - 1))
    axes_to_perm["classifier.1.bias"] = ("P_Dense_0",)
    axes_to_perm["classifier.4.weight"] = ("P_Dense_1", "P_Dense_0")
    axes_to_perm["classifier.4.bias"] = ("P_Dense_1",)
    axes_to_perm["classifier.6.weight"] = (None, "P_Dense_1")
    axes_to_perm["classifier.6.bias"] = (None,)
    return permutation_spec_from_axes_to_perm(axes_to_perm)


def build_layer_blocks_activation_module_map(block_conv_counts: list[int]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    conv_idx = 0
    for block_idx, num_layers in enumerate(block_conv_counts):
        for layer_idx in range(num_layers):
            mapping[f"P_Conv_{conv_idx}"] = f"activation_blocks.{block_idx}.{layer_idx}"
            conv_idx += 1
    mapping["P_Dense_0"] = "classifier.2"
    mapping["P_Dense_1"] = "classifier.5"
    return mapping


def build_features_activation_module_map(block_conv_counts: list[int]) -> Dict[str, str]:
    _, relu_indices = external_conv_relu_indices(block_conv_counts)
    mapping = {f"P_Conv_{index}": f"features.{relu_index}" for index, relu_index in enumerate(relu_indices)}
    mapping["P_Dense_0"] = "classifier.2"
    mapping["P_Dense_1"] = "classifier.5"
    return mapping


@dataclass
class VGGExternalRuntime:
    layout: str
    vgg_name: str
    block_conv_counts: list[int]
    perm_spec: PermutationSpec
    activation_module_map: Dict[str, str]
    build_model: Callable[[torch.device | None], torch.nn.Module]


def build_runtime(layout: str, state_dict: Dict[str, torch.Tensor]) -> VGGExternalRuntime:
    if layout == "layer_blocks":
        vgg_name = infer_layer_blocks_vgg_name(state_dict)
        block_conv_counts = ARCH_BLOCK_CONV_COUNTS[vgg_name]
        if vgg_name != "VGG16":
            raise ValueError(
                f"layer_blocks layout currently only supports VGG16-compatible checkpoints, got inferred {vgg_name}."
            )
        return VGGExternalRuntime(
            layout=layout,
            vgg_name=vgg_name,
            block_conv_counts=block_conv_counts,
            perm_spec=vgg16_permutation_spec() if block_conv_counts == [2, 2, 3, 3, 3] else build_layer_blocks_permutation_spec(block_conv_counts),
            activation_module_map=build_layer_blocks_activation_module_map(block_conv_counts),
            build_model=lambda device=None: create_vgg16_model(num_classes=10, device=device),
        )
    if layout == "features":
        vgg_name = infer_external_vgg_name(state_dict)
        block_conv_counts = ARCH_BLOCK_CONV_COUNTS[vgg_name]
        return VGGExternalRuntime(
            layout=layout,
            vgg_name=vgg_name,
            block_conv_counts=block_conv_counts,
            perm_spec=vgg16_features_permutation_spec() if block_conv_counts == [2, 2, 3, 3, 3] else build_features_permutation_spec(block_conv_counts),
            activation_module_map=build_features_activation_module_map(block_conv_counts),
            build_model=lambda device=None, *, _vgg_name=vgg_name: build_external_vgg_model(_vgg_name, device),
        )
    raise ValueError(f"Unsupported layout={layout!r}")


def load_model_for_runtime(
    checkpoint_path: str,
    *,
    runtime: VGGExternalRuntime,
    device: torch.device,
) -> torch.nn.Module:
    if runtime.layout == "layer_blocks":
        model = load_vgg16_model(checkpoint_path, map_location="cpu").to(device)
        model.eval()
        return model

    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = extract_state_dict(payload)
    model = runtime.build_model(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


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
    if x.ndim == 2:
        return x
    if x.ndim == 4:
        return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])
    raise ValueError(f"Unsupported activation shape: {tuple(x.shape)}")


def subsample_rows(
    a: torch.Tensor,
    b: torch.Tensor,
    max_rows: int,
    rng: np.random.RandomState,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    module_name_map: Dict[str, str],
) -> Dict[str, np.ndarray]:
    tap_a = ActivationTap(model_a, module_name_map)
    tap_b = ActivationTap(model_b, module_name_map)
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

                for layer_key in module_name_map:
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
    for layer_key in module_name_map:
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
    *,
    perm_spec: PermutationSpec,
    max_iter: int,
    seed: int,
) -> Dict[str, np.ndarray]:
    params_a = state_dict_to_perm_params(state_a, perm_spec)
    params_b = state_dict_to_perm_params(state_b, perm_spec)
    return weight_matching(
        ps=perm_spec,
        params_a=params_a,
        params_b=params_b,
        max_iter=max_iter,
        seed=seed,
        silent=False,
    )


def apply_permutation_to_state(
    state_dict: OrderedDict[str, torch.Tensor],
    permutation: Dict[str, np.ndarray],
    *,
    perm_spec: PermutationSpec,
) -> OrderedDict[str, torch.Tensor]:
    params = state_dict_to_perm_params(state_dict, perm_spec)
    params_aligned = apply_permutation(perm_spec, permutation, params)

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


def evaluate_interpolation_curves(
    *,
    state_a: OrderedDict[str, torch.Tensor],
    state_b: OrderedDict[str, torch.Tensor],
    loaders,
    build_model_fn: Callable[[torch.device | None], torch.nn.Module],
    device: torch.device,
    num_points: int,
    max_batches: int,
) -> Dict[str, object]:
    interp_model = build_model_fn(device)
    ts = np.linspace(0.0, 1.0, num_points, dtype=np.float64)

    train_loss: list[float] = []
    train_acc: list[float] = []
    test_loss: list[float] = []
    test_acc: list[float] = []

    for t in ts:
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = ((1.0 - t) * state_a[key].detach().cpu() + t * state_b[key].detach().cpu())
        interp_model.load_state_dict(interp_state)

        train_metrics = evaluate_model_limited(interp_model, loaders["train"], device, max_batches=max_batches)
        test_metrics = evaluate_model_limited(interp_model, loaders["test"], device, max_batches=max_batches)

        train_loss.append(float(train_metrics["loss"]))
        train_acc.append(float(train_metrics["accuracy"]))
        test_loss.append(float(test_metrics["loss"]))
        test_acc.append(float(test_metrics["accuracy"]))

    endpoint_avg_test_loss = 0.5 * (test_loss[0] + test_loss[-1])
    max_test_loss = float(max(test_loss))
    min_test_acc = float(min(test_acc))
    endpoint_avg_train_loss = 0.5 * (train_loss[0] + train_loss[-1])
    max_train_loss = float(max(train_loss))
    min_train_acc = float(min(train_acc))

    return {
        "t": ts.tolist(),
        "train_loss": train_loss,
        "train_acc": train_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "train_barrier": max_train_loss - endpoint_avg_train_loss,
        "max_train_loss": max_train_loss,
        "endpoint_avg_train_loss": endpoint_avg_train_loss,
        "min_train_acc": min_train_acc,
        "test_barrier": max_test_loss - endpoint_avg_test_loss,
        "max_test_loss": max_test_loss,
        "endpoint_avg_test_loss": endpoint_avg_test_loss,
        "min_test_acc": min_test_acc,
    }


def get_first_batch(loader):
    for batch in loader:
        return batch
    raise RuntimeError("Loader produced no batches")


def compare_model_outputs(
    model_original: torch.nn.Module,
    model_permuted: torch.nn.Module,
    batch,
    *,
    device: torch.device,
    atol: float,
    rtol: float,
) -> Dict[str, float | int | bool]:
    inputs, _ = batch
    inputs = inputs.to(device)

    model_original.eval()
    model_permuted.eval()
    with torch.no_grad():
        outputs_original = model_original(inputs)
        outputs_permuted = model_permuted(inputs)

    diff = torch.abs(outputs_original - outputs_permuted)
    argmax_match = (outputs_original.argmax(dim=1) == outputs_permuted.argmax(dim=1)).float().mean().item()

    return {
        "batch_size": int(inputs.shape[0]),
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "allclose": bool(torch.allclose(outputs_original, outputs_permuted, atol=atol, rtol=rtol)),
        "same_argmax_fraction": float(argmax_match),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def plot_before_after_curves(
    *,
    x: list[float],
    y_before: list[float],
    y_after: list[float],
    title: str,
    ylabel: str,
    output_path: str,
) -> None:
    plt.figure()
    plt.plot(x, y_before, label="Before Alignment", color="tab:gray", linewidth=2.0)
    plt.plot(x, y_after, label="After Alignment", color="tab:orange", linewidth=2.0)
    plt.xlabel("t (interpolation parameter)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.5)
    plt.legend()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Permutation alignment for VGG CIFAR-10 checkpoints.")
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
    parser.add_argument("--num-eval-points", type=int, default=21, help="Interpolation points for saved curves")
    parser.add_argument("--eval-max-batches", type=int, default=0, help="Max train/test batches per interpolation point; <=0 uses full split")
    parser.add_argument("--lmc-threshold", type=float, default=0.1, help="Barrier threshold for LMC")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for activation row subsampling")
    parser.add_argument("--functional-atol", type=float, default=1e-5, help="Absolute tolerance for functional equivalence")
    parser.add_argument("--functional-rtol", type=float, default=1e-4, help="Relative tolerance for functional equivalence")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_a_payload = torch.load(args.model_a, map_location="cpu")
    model_b_payload = torch.load(args.model_b, map_location="cpu")
    state_a = extract_state_dict(model_a_payload)
    state_b = extract_state_dict(model_b_payload)

    layout_a = infer_layout(state_a)
    layout_b = infer_layout(state_b)
    if layout_a != layout_b:
        raise ValueError(f"Checkpoint layouts do not match: model_a={layout_a}, model_b={layout_b}")

    runtime = build_runtime(layout_a, state_a)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Detected checkpoint layout: {runtime.layout}")
    print(f"Detected VGG variant: {runtime.vgg_name}")
    print(f"Loading checkpoints:\n  A={args.model_a}\n  B={args.model_b}")

    model_a = load_model_for_runtime(args.model_a, runtime=runtime, device=device)
    model_b = load_model_for_runtime(args.model_b, runtime=runtime, device=device)

    loaders, _num_classes = load_cifar10_eval_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

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
            module_name_map=runtime.activation_module_map,
        )
    else:
        print("\nComputing weight-matching permutation...")
        perm = compute_weight_permutation(
            state_a=state_a,
            state_b=state_b,
            perm_spec=runtime.perm_spec,
            max_iter=args.wm_max_iter,
            seed=args.seed,
        )

    print("\nApplying permutation to model B...")
    state_b_aligned = apply_permutation_to_state(
        state_b,
        perm,
        perm_spec=runtime.perm_spec,
    )

    model_b_aligned = runtime.build_model(device)
    model_b_aligned.load_state_dict(state_b_aligned)
    model_b_aligned.eval()

    ckpt_b = torch.load(args.model_b, map_location="cpu")
    aligned_state_cpu = OrderedDict((k, v.detach().cpu()) for k, v in state_b_aligned.items())
    if isinstance(ckpt_b, dict) and "model_state" in ckpt_b:
        ckpt_b["model_state"] = aligned_state_cpu
    elif isinstance(ckpt_b, dict) and "state_dict" in ckpt_b:
        ckpt_b["state_dict"] = aligned_state_cpu
    elif isinstance(ckpt_b, dict):
        ckpt_b = dict(aligned_state_cpu)
    else:
        ckpt_b = {"model_state": aligned_state_cpu}

    method_tag = "activation-matched" if args.method == "activation_matching" else "weight-matched"
    aligned_ckpt_path = os.path.join(args.output_dir, f"checkpoint-200-{method_tag}.pt")
    torch.save(ckpt_b, aligned_ckpt_path)
    print(f"Aligned checkpoint saved to: {aligned_ckpt_path}")

    perm_npz_path = os.path.join(args.output_dir, f"{args.method}_permutation.npz")
    np.savez(perm_npz_path, **perm)
    print(f"Permutation saved to: {perm_npz_path}")

    print("\nChecking functional equivalence of model B before/after permutation...")
    train_batch = get_first_batch(loaders["train"])
    test_batch = get_first_batch(loaders["test"])
    functional_equivalence = {
        "layout": runtime.layout,
        "train_batch": compare_model_outputs(
            model_b,
            model_b_aligned,
            train_batch,
            device=device,
            atol=args.functional_atol,
            rtol=args.functional_rtol,
        ),
        "test_batch": compare_model_outputs(
            model_b,
            model_b_aligned,
            test_batch,
            device=device,
            atol=args.functional_atol,
            rtol=args.functional_rtol,
        ),
        "train_eval_original": evaluate_model_limited(model_b, loaders["train"], device, max_batches=args.eval_max_batches),
        "train_eval_permuted": evaluate_model_limited(model_b_aligned, loaders["train"], device, max_batches=args.eval_max_batches),
        "test_eval_original": evaluate_model_limited(model_b, loaders["test"], device, max_batches=args.eval_max_batches),
        "test_eval_permuted": evaluate_model_limited(model_b_aligned, loaders["test"], device, max_batches=args.eval_max_batches),
    }
    functional_equivalence["train_eval_delta"] = {
        "loss": functional_equivalence["train_eval_permuted"]["loss"] - functional_equivalence["train_eval_original"]["loss"],
        "accuracy": functional_equivalence["train_eval_permuted"]["accuracy"] - functional_equivalence["train_eval_original"]["accuracy"],
    }
    functional_equivalence["test_eval_delta"] = {
        "loss": functional_equivalence["test_eval_permuted"]["loss"] - functional_equivalence["test_eval_original"]["loss"],
        "accuracy": functional_equivalence["test_eval_permuted"]["accuracy"] - functional_equivalence["test_eval_original"]["accuracy"],
    }
    functional_equivalence_path = os.path.join(args.output_dir, "functional_equivalence.json")
    save_json(functional_equivalence_path, functional_equivalence)
    print(f"Functional equivalence report saved to: {functional_equivalence_path}")

    print("\nEvaluating interpolation curves before and after alignment...")
    curves_before = evaluate_interpolation_curves(
        state_a=state_a,
        state_b=state_b,
        loaders=loaders,
        build_model_fn=runtime.build_model,
        device=device,
        num_points=args.num_eval_points,
        max_batches=args.eval_max_batches,
    )
    curves_after = evaluate_interpolation_curves(
        state_a=state_a,
        state_b=state_b_aligned,
        loaders=loaders,
        build_model_fn=runtime.build_model,
        device=device,
        num_points=args.num_eval_points,
        max_batches=args.eval_max_batches,
    )

    interpolation_curves = {
        "num_eval_points": int(args.num_eval_points),
        "eval_max_batches": int(args.eval_max_batches),
        "before_alignment": curves_before,
        "after_alignment": curves_after,
    }
    interpolation_curves_path = os.path.join(args.output_dir, "interpolation_curves.json")
    save_json(interpolation_curves_path, interpolation_curves)
    print(f"Interpolation curves saved to: {interpolation_curves_path}")

    plot_before_after_curves(
        x=curves_before["t"],
        y_before=curves_before["test_loss"],
        y_after=curves_after["test_loss"],
        title=f"{runtime.vgg_name}: test loss",
        ylabel="Test Loss",
        output_path=os.path.join(args.output_dir, "compare_test_loss.png"),
    )
    plot_before_after_curves(
        x=curves_before["t"],
        y_before=curves_before["test_acc"],
        y_after=curves_after["test_acc"],
        title=f"{runtime.vgg_name}: test accuracy",
        ylabel="Accuracy (%)",
        output_path=os.path.join(args.output_dir, "compare_test_accuracy.png"),
    )
    plot_before_after_curves(
        x=curves_before["t"],
        y_before=curves_before["train_loss"],
        y_after=curves_after["train_loss"],
        title=f"{runtime.vgg_name}: train loss",
        ylabel="Train Loss",
        output_path=os.path.join(args.output_dir, "compare_train_loss.png"),
    )
    plot_before_after_curves(
        x=curves_before["t"],
        y_before=curves_before["train_acc"],
        y_after=curves_after["train_acc"],
        title=f"{runtime.vgg_name}: train accuracy",
        ylabel="Accuracy (%)",
        output_path=os.path.join(args.output_dir, "compare_train_accuracy.png"),
    )

    dist_before = compute_state_dict_l2_distance(state_a, state_b)
    dist_after = compute_state_dict_l2_distance(state_a, state_b_aligned)

    summary = {
        "model_a": args.model_a,
        "model_b": args.model_b,
        "aligned_model_b": aligned_ckpt_path,
        "method": args.method,
        "layout": runtime.layout,
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
            "splits": ["train", "test"],
            "num_eval_points": args.num_eval_points,
            "max_eval_batches": args.eval_max_batches,
        },
        "lmc_threshold": args.lmc_threshold,
        "functional_equivalence_path": functional_equivalence_path,
        "interpolation_curves_path": interpolation_curves_path,
        "before_alignment": {
            "distance": dist_before,
            "barrier": {
                "t": curves_before["t"],
                "train_loss": curves_before["train_loss"],
                "train_acc": curves_before["train_acc"],
                "test_loss": curves_before["test_loss"],
                "test_acc": curves_before["test_acc"],
                "train_barrier": curves_before["train_barrier"],
                "max_train_loss": curves_before["max_train_loss"],
                "endpoint_avg_train_loss": curves_before["endpoint_avg_train_loss"],
                "min_train_acc": curves_before["min_train_acc"],
                "barrier": curves_before["test_barrier"],
                "max_test_loss": curves_before["max_test_loss"],
                "endpoint_avg_test_loss": curves_before["endpoint_avg_test_loss"],
                "min_test_acc": curves_before["min_test_acc"],
            },
            "is_lmc": curves_before["test_barrier"] < args.lmc_threshold,
        },
        "after_alignment": {
            "distance": dist_after,
            "barrier": {
                "t": curves_after["t"],
                "train_loss": curves_after["train_loss"],
                "train_acc": curves_after["train_acc"],
                "test_loss": curves_after["test_loss"],
                "test_acc": curves_after["test_acc"],
                "train_barrier": curves_after["train_barrier"],
                "max_train_loss": curves_after["max_train_loss"],
                "endpoint_avg_train_loss": curves_after["endpoint_avg_train_loss"],
                "min_train_acc": curves_after["min_train_acc"],
                "barrier": curves_after["test_barrier"],
                "max_test_loss": curves_after["max_test_loss"],
                "endpoint_avg_test_loss": curves_after["endpoint_avg_test_loss"],
                "min_test_acc": curves_after["min_test_acc"],
            },
            "is_lmc": curves_after["test_barrier"] < args.lmc_threshold,
        },
        "improvement": {
            "barrier_delta": curves_before["test_barrier"] - curves_after["test_barrier"],
            "barrier_relative_reduction_percent": (
                100.0 * (curves_before["test_barrier"] - curves_after["test_barrier"]) / curves_before["test_barrier"]
            )
            if abs(curves_before["test_barrier"]) > 1e-12
            else 0.0,
            "l2_delta": dist_before["l2_distance"] - dist_after["l2_distance"],
        },
    }

    summary_path = os.path.join(args.output_dir, f"{args.method}_lmc_summary.json")
    save_json(summary_path, summary)
    print(f"Summary saved to: {summary_path}")

    print("\n=== Final metrics ===")
    print(
        "Before alignment: "
        f"barrier={curves_before['test_barrier']:.6f}, "
        f"min_test_acc={curves_before['min_test_acc']:.2f}%"
    )
    print(
        "After alignment:  "
        f"barrier={curves_after['test_barrier']:.6f}, "
        f"min_test_acc={curves_after['min_test_acc']:.2f}%"
    )
    print(
        f"LMC status (threshold={args.lmc_threshold:.4f}): "
        f"before={summary['before_alignment']['is_lmc']}, "
        f"after={summary['after_alignment']['is_lmc']}"
    )


if __name__ == "__main__":
    main()
