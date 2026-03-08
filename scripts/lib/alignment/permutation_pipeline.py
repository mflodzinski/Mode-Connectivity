"""Shared helpers for the VGG16 permutation-path alignment pipeline."""

from __future__ import annotations

import copy
import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from scripts.lib.analysis.alignment import (
    create_vgg16_model,
    evaluate_model,
    load_cifar10_eval_loaders,
)
from scripts.lib.core.checkpoint import load_checkpoint, load_state_dict
from scripts.lib.core.output import ResultSaver, save_json
from scripts.lib.transform.random_permutation import VGG16RandomPermutation


BASELINE_DISPLAY_NAMES = {
    "baseline_1_no_permutation": "Baseline 1: No permutation",
    "baseline_2_c2m3_direct": "Baseline 2: C2M3 direct endpoint matching",
    "baseline_3_greedy_adjacent": "Baseline 3: Greedy adjacent path matching",
    "baseline_4_c2m3_global": "Baseline 4: Global multi-checkpoint C2M3",
}


def resolve_device(device: str) -> torch.device:
    """Resolve ``auto`` into an actual torch device."""

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def load_vgg16_cifar10_loaders(data_path: str, batch_size: int, num_workers: int):
    """Load the CIFAR10 train/test eval loaders used for interpolation checks."""

    return load_cifar10_eval_loaders(data_path=data_path, batch_size=batch_size, num_workers=num_workers)


def load_checkpoint_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a checkpoint and return its state dict."""

    return load_state_dict(path)


def state_dict_l2_distance(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> float:
    """Compute the Euclidean distance between two state dicts."""

    if set(state_a.keys()) != set(state_b.keys()):
        raise ValueError("State dict keys do not match for L2 distance computation.")

    sq_sum = torch.zeros((), dtype=torch.float64)
    for key in state_a:
        delta = state_a[key].detach().cpu().to(torch.float64) - state_b[key].detach().cpu().to(torch.float64)
        sq_sum = sq_sum + torch.sum(delta * delta)
    return float(torch.sqrt(sq_sum).item())


def identity_permutation(state_dict: Dict[str, torch.Tensor], permutation_spec) -> Dict[str, np.ndarray]:
    """Create an identity permutation matching the local permutation spec."""

    identity = {}
    for perm_name, axes in permutation_spec.perm_to_axes.items():
        param_name, axis = axes[0]
        identity[perm_name] = np.arange(state_dict[param_name].shape[axis], dtype=np.int64)
    return identity


def state_dict_to_perm_params(
    state_dict: Dict[str, torch.Tensor],
    permutation_spec,
) -> Dict[str, torch.Tensor]:
    """Filter a state dict down to the parameter keys referenced by a permutation spec."""

    return {key: value for key, value in state_dict.items() if key in permutation_spec.axes_to_perm}


def to_numpy_permutation(permutation: Dict[str, np.ndarray | torch.Tensor | List[int]]) -> Dict[str, np.ndarray]:
    """Normalize a permutation dictionary to numpy index arrays."""

    normalized = {}
    for key, value in permutation.items():
        if isinstance(value, torch.Tensor):
            normalized[key] = value.detach().cpu().numpy().astype(np.int64)
        else:
            normalized[key] = np.asarray(value, dtype=np.int64)
    return normalized


def serialize_permutation(permutation: Dict[str, np.ndarray | torch.Tensor | List[int]]) -> Dict[str, List[int]]:
    """Convert a permutation dictionary into JSON-friendly lists."""

    return {key: to_numpy_permutation(permutation)[key].tolist() for key in permutation}


def convert_perm_keys_to_apply_format(permutation: Dict[str, np.ndarray | torch.Tensor | List[int]]) -> Dict[str, np.ndarray]:
    """Convert ``P_Conv_*``/``P_Dense_*`` keys into the apply-format keys used locally."""

    converted = {}
    for key, value in to_numpy_permutation(permutation).items():
        if key.startswith("P_Conv_"):
            converted[f"conv_{key[7:]}"] = value
        elif key.startswith("P_Dense_"):
            converted[f"fc_{key[8:]}"] = value
        else:
            converted[key] = value
    return converted


def convert_apply_keys_to_perm_format(permutation: Dict[str, np.ndarray | torch.Tensor | List[int]]) -> Dict[str, np.ndarray]:
    """Convert apply-format ``conv_*``/``fc_*`` keys back into ``P_*`` keys."""

    converted = {}
    for key, value in to_numpy_permutation(permutation).items():
        if key.startswith("conv_"):
            converted[f"P_Conv_{key[5:]}"] = value
        elif key.startswith("fc_"):
            converted[f"P_Dense_{key[3:]}"] = value
        else:
            converted[key] = value
    return converted


def perm_indices_to_matrix(perm_indices: np.ndarray | torch.Tensor | List[int]) -> np.ndarray:
    """Convert an index permutation into its permutation-matrix representation."""

    indices = np.asarray(perm_indices, dtype=np.int64)
    return np.eye(len(indices), dtype=np.float64)[indices]


def perm_matrix_to_indices(perm_matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert a permutation matrix back into index format."""

    matrix = np.asarray(perm_matrix)
    return matrix.argmax(axis=1).astype(np.int64)


def compose_permutations(
    left: Dict[str, np.ndarray | torch.Tensor | List[int]],
    right: Dict[str, np.ndarray | torch.Tensor | List[int]],
) -> Dict[str, np.ndarray]:
    """Compose ``right`` followed by ``left`` in permutation-matrix form."""

    left_np = to_numpy_permutation(left)
    right_np = to_numpy_permutation(right)
    composed = {}
    for key in left_np:
        composed[key] = perm_matrix_to_indices(perm_indices_to_matrix(left_np[key]) @ perm_indices_to_matrix(right_np[key]))
    return composed


def compose_permutation_sequence(
    permutations: Sequence[Dict[str, np.ndarray | torch.Tensor | List[int]]],
) -> Dict[str, np.ndarray]:
    """Compose a sequence of layerwise permutations from left to right."""

    if not permutations:
        raise ValueError("At least one permutation is required for composition.")
    composed = to_numpy_permutation(permutations[0])
    for permutation in permutations[1:]:
        composed = compose_permutations(composed, permutation)
    return composed


def unfactor_factored_permutations(
    factored_permutations: Dict[str, Dict[str, np.ndarray | torch.Tensor | List[int]]],
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Expand factored per-symbol permutations into all pairwise endpoint permutations."""

    symbols = sorted(factored_permutations.keys())
    factored_np = {symbol: to_numpy_permutation(perms) for symbol, perms in factored_permutations.items()}
    unfactored = {source: {} for source in symbols}

    for source in symbols:
        for target in symbols:
            if source == target:
                continue
            unfactored[source][target] = {}
            for perm_name in factored_np[source]:
                source_matrix = perm_indices_to_matrix(factored_np[source][perm_name])
                target_matrix = perm_indices_to_matrix(factored_np[target][perm_name])
                unfactored[source][target][perm_name] = perm_matrix_to_indices(source_matrix @ target_matrix.T)
    return unfactored


def derive_endpoint_permutation_from_factored(
    factored_permutations: Dict[str, Dict[str, np.ndarray | torch.Tensor | List[int]]],
    *,
    fixed_symbol: str,
    permutee_symbol: str,
) -> Dict[str, np.ndarray]:
    """Derive the endpoint permutation ``permutee -> fixed`` from factored global permutations."""

    factored_np = {symbol: to_numpy_permutation(perms) for symbol, perms in factored_permutations.items()}
    derived = {}
    for perm_name in factored_np[fixed_symbol]:
        fixed_matrix = perm_indices_to_matrix(factored_np[fixed_symbol][perm_name])
        permutee_matrix = perm_indices_to_matrix(factored_np[permutee_symbol][perm_name])
        derived[perm_name] = perm_matrix_to_indices(fixed_matrix @ permutee_matrix.T)
    return derived


def apply_endpoint_permutation_to_state_dict(
    state_dict: Dict[str, torch.Tensor],
    permutation: Dict[str, np.ndarray | torch.Tensor | List[int]],
) -> OrderedDict:
    """Apply a VGG16 endpoint permutation to a state dict."""

    perm_gen = VGG16RandomPermutation()
    converted = convert_perm_keys_to_apply_format(permutation)
    original = OrderedDict((key, value.clone()) for key, value in state_dict.items())
    return perm_gen.apply_to_state_dict(original, converted)


def save_checkpoint_with_state_dict(
    reference_checkpoint_path: str,
    output_path: str,
    state_dict: Dict[str, torch.Tensor],
    *,
    metadata: Dict | None = None,
) -> None:
    """Save a checkpoint that preserves the original checkpoint wrapper when possible."""

    checkpoint = load_checkpoint(reference_checkpoint_path)
    if not isinstance(checkpoint, dict):
        checkpoint = {}

    updated = copy.deepcopy(checkpoint)
    updated["model_state"] = OrderedDict((key, value.detach().cpu()) for key, value in state_dict.items())
    if metadata is not None:
        updated["alignment_metadata"] = metadata

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(updated, output)


def evaluate_linear_interpolation(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
    loaders,
    *,
    num_points: int,
    device: torch.device,
    num_classes: int = 10,
) -> Dict[str, np.ndarray]:
    """Evaluate linear interpolation between two VGG16 state dicts."""

    interp_model = create_vgg16_model(num_classes=num_classes, device=device)
    ts = np.linspace(0.0, 1.0, num_points)

    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for t in ts:
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = ((1.0 - t) * state_a[key] + t * state_b[key]).detach().cpu()

        interp_model.load_state_dict(interp_state)
        train_metrics = evaluate_model(interp_model, loaders["train"], device)
        test_metrics = evaluate_model(interp_model, loaders["test"], device)

        train_loss.append(train_metrics["loss"])
        train_acc.append(train_metrics["accuracy"])
        test_loss.append(test_metrics["loss"])
        test_acc.append(test_metrics["accuracy"])

    results = {
        "ts": ts,
        "tr_loss": np.asarray(train_loss, dtype=np.float64),
        "tr_acc": np.asarray(train_acc, dtype=np.float64),
        "tr_err": 100.0 - np.asarray(train_acc, dtype=np.float64),
        "te_loss": np.asarray(test_loss, dtype=np.float64),
        "te_acc": np.asarray(test_acc, dtype=np.float64),
        "te_err": 100.0 - np.asarray(test_acc, dtype=np.float64),
    }
    return results


def save_interpolation_results(output_path: str, results: Dict[str, np.ndarray]) -> None:
    """Save interpolation arrays in the standard `.npz` format used by this repo."""

    ResultSaver.save_standard(
        output_path,
        results["ts"],
        {
            "loss": results["tr_loss"],
            "acc": results["tr_acc"],
            "err": results["tr_err"],
        },
        {
            "loss": results["te_loss"],
            "acc": results["te_acc"],
            "err": results["te_err"],
        },
    )


def compute_barrier_metrics(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute the requested interpolation barrier metrics."""

    tr_loss = results["tr_loss"]
    te_loss = results["te_loss"]
    tr_acc = results["tr_acc"]
    te_acc = results["te_acc"]

    metrics = {
        "train_loss_barrier_avg": float(np.max(tr_loss) - 0.5 * (tr_loss[0] + tr_loss[-1])),
        "test_loss_barrier_avg": float(np.max(te_loss) - 0.5 * (te_loss[0] + te_loss[-1])),
        "train_loss_barrier_max_endpoint": float(np.max(tr_loss) - max(tr_loss[0], tr_loss[-1])),
        "test_loss_barrier_max_endpoint": float(np.max(te_loss) - max(te_loss[0], te_loss[-1])),
        "min_train_acc": float(np.min(tr_acc)),
        "min_test_acc": float(np.min(te_acc)),
        "train_acc_drop_from_endpoint_min": float(min(tr_acc[0], tr_acc[-1]) - np.min(tr_acc)),
        "test_acc_drop_from_endpoint_min": float(min(te_acc[0], te_acc[-1]) - np.min(te_acc)),
    }
    return metrics


def get_test_batch(loaders, batch_index: int = 0):
    """Fetch a deterministic batch from the test loader."""

    for index, batch in enumerate(loaders["test"]):
        if index == batch_index:
            return batch
    raise IndexError(f"Requested test batch {batch_index}, but the loader exhausted early.")


def verify_functional_equivalence(
    original_state: Dict[str, torch.Tensor],
    permuted_state: Dict[str, torch.Tensor],
    batch,
    *,
    device: torch.device,
    atol: float,
    rtol: float,
    num_classes: int = 10,
    permutation_applied: bool = True,
) -> Dict[str, float | int | bool]:
    """Compare model outputs on a single batch before and after applying a permutation."""

    inputs, _ = batch
    inputs = inputs.to(device)

    model_original = create_vgg16_model(num_classes=num_classes, device=device)
    model_permuted = create_vgg16_model(num_classes=num_classes, device=device)
    model_original.load_state_dict(original_state)
    model_permuted.load_state_dict(permuted_state)
    model_original.eval()
    model_permuted.eval()

    with torch.no_grad():
        outputs_original = model_original(inputs)
        outputs_permuted = model_permuted(inputs)

    diff = torch.abs(outputs_original - outputs_permuted)
    argmax_match = (outputs_original.argmax(dim=1) == outputs_permuted.argmax(dim=1)).float().mean().item()

    return {
        "permutation_applied": permutation_applied,
        "batch_size": int(inputs.shape[0]),
        "max_abs_logit_diff": float(diff.max().item()),
        "mean_abs_logit_diff": float(diff.mean().item()),
        "allclose": bool(torch.allclose(outputs_original, outputs_permuted, atol=atol, rtol=rtol)),
        "same_argmax_fraction": float(argmax_match),
        "atol": float(atol),
        "rtol": float(rtol),
    }


def write_json(path: str, payload: Dict) -> None:
    """Write JSON to disk with parent directory creation."""

    save_json(payload, path, indent=2)


def write_summary_files(output_dir: str, rows: Sequence[Dict]) -> None:
    """Write CSV, JSON, and Markdown summary tables for all baselines."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    json_path = output_path / "comparison.json"
    csv_path = output_path / "comparison.csv"
    markdown_path = output_path / "comparison.md"

    save_json(list(rows), json_path, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    with open(markdown_path, "w") as handle:
        if not fieldnames:
            handle.write("| baseline |\n| --- |\n")
            return

        handle.write("| " + " | ".join(fieldnames) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(fieldnames)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row[field]) for field in fieldnames) + " |\n")


def build_summary_row(
    baseline_key: str,
    barrier_metrics: Dict[str, float],
    equivalence_metrics: Dict[str, float | int | bool],
) -> Dict[str, float | int | bool | str]:
    """Flatten per-baseline metrics into one summary row."""

    return {
        "baseline_key": baseline_key,
        "baseline_name": BASELINE_DISPLAY_NAMES[baseline_key],
        "test_loss_barrier_avg": barrier_metrics["test_loss_barrier_avg"],
        "test_loss_barrier_max_endpoint": barrier_metrics["test_loss_barrier_max_endpoint"],
        "min_test_acc": barrier_metrics["min_test_acc"],
        "test_acc_drop_from_endpoint_min": barrier_metrics["test_acc_drop_from_endpoint_min"],
        "train_loss_barrier_avg": barrier_metrics["train_loss_barrier_avg"],
        "train_loss_barrier_max_endpoint": barrier_metrics["train_loss_barrier_max_endpoint"],
        "min_train_acc": barrier_metrics["min_train_acc"],
        "train_acc_drop_from_endpoint_min": barrier_metrics["train_acc_drop_from_endpoint_min"],
        "max_abs_logit_diff": equivalence_metrics["max_abs_logit_diff"],
        "mean_abs_logit_diff": equivalence_metrics["mean_abs_logit_diff"],
        "same_argmax_fraction": equivalence_metrics["same_argmax_fraction"],
        "allclose": equivalence_metrics["allclose"],
    }


def write_permutation_json(path: str, permutation: Dict[str, np.ndarray | torch.Tensor | List[int]]) -> None:
    """Save a permutation dictionary as JSON."""

    write_json(path, serialize_permutation(permutation))
