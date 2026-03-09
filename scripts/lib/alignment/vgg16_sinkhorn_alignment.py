"""VGG16-specific Sinkhorn alignment prototype with optional positive scaling.

This module intentionally stays architecture-specific. The goal is a minimal
research prototype for comparing:
- no alignment
- Sinkhorn permutation-only alignment
- Sinkhorn + positive diagonal scaling alignment

The math-to-code mapping follows the VGG16 hidden-layer symmetries directly.
For each hidden layer l we learn:
    S_l = Sinkhorn(Z_l / tau)
    D_l = diag(exp(u_l))
and use:
    M_l = S_l D_l

For input transport into the next layer we avoid inverting dense soft matrices.
Instead we use the same soft convention as the permutation-only baseline:
applying `Q_l = S_l D_l^{-1}` on the input axis corresponds to right-multiplying
the layer weight by `D_l^{-1} S_l^T`.
"""

from __future__ import annotations

import itertools
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.func import functional_call
from torch.utils.data import DataLoader, Subset

from scripts.lib.analysis.alignment import create_vgg16_model, load_cifar10_eval_loaders
from scripts.lib.alignment.permutation_pipeline import (
    load_checkpoint_state_dict,
    resolve_device,
    save_checkpoint_with_state_dict,
    write_json,
)
from scripts.lib.core.output import ensure_dir


MethodName = str

METHOD_PERM_ONLY = "perm_only"
METHOD_PERM_SCALE = "perm_scale"
VALID_METHODS = {METHOD_PERM_ONLY, METHOD_PERM_SCALE}


@dataclass(frozen=True)
class HiddenLayerSpec:
    """Metadata for one permutable VGG16 hidden layer."""

    perm_name: str
    weight_key: str
    bias_key: str
    size: int
    layer_type: str


@dataclass
class AlignmentMatrices:
    """Soft or hard monomial transport objects for all hidden layers."""

    permutations: OrderedDict[str, torch.Tensor]
    output_monomials: OrderedDict[str, torch.Tensor]
    input_transports: OrderedDict[str, torch.Tensor]
    log_scales: OrderedDict[str, torch.Tensor]
    scales: OrderedDict[str, torch.Tensor]


@dataclass
class AlignmentResult:
    """Filesystem outputs for one trained alignment method."""

    method: str
    method_dir: str
    soft_checkpoint_path: str
    hard_checkpoint_path: str
    artifact_path: str
    history_path: str
    metadata_path: str


VGG16_HIDDEN_LAYER_SPECS: tuple[HiddenLayerSpec, ...] = (
    HiddenLayerSpec("P_Conv_0", "layer_blocks.0.0.weight", "layer_blocks.0.0.bias", 64, "conv"),
    HiddenLayerSpec("P_Conv_1", "layer_blocks.0.1.weight", "layer_blocks.0.1.bias", 64, "conv"),
    HiddenLayerSpec("P_Conv_2", "layer_blocks.1.0.weight", "layer_blocks.1.0.bias", 128, "conv"),
    HiddenLayerSpec("P_Conv_3", "layer_blocks.1.1.weight", "layer_blocks.1.1.bias", 128, "conv"),
    HiddenLayerSpec("P_Conv_4", "layer_blocks.2.0.weight", "layer_blocks.2.0.bias", 256, "conv"),
    HiddenLayerSpec("P_Conv_5", "layer_blocks.2.1.weight", "layer_blocks.2.1.bias", 256, "conv"),
    HiddenLayerSpec("P_Conv_6", "layer_blocks.2.2.weight", "layer_blocks.2.2.bias", 256, "conv"),
    HiddenLayerSpec("P_Conv_7", "layer_blocks.3.0.weight", "layer_blocks.3.0.bias", 512, "conv"),
    HiddenLayerSpec("P_Conv_8", "layer_blocks.3.1.weight", "layer_blocks.3.1.bias", 512, "conv"),
    HiddenLayerSpec("P_Conv_9", "layer_blocks.3.2.weight", "layer_blocks.3.2.bias", 512, "conv"),
    HiddenLayerSpec("P_Conv_10", "layer_blocks.4.0.weight", "layer_blocks.4.0.bias", 512, "conv"),
    HiddenLayerSpec("P_Conv_11", "layer_blocks.4.1.weight", "layer_blocks.4.1.bias", 512, "conv"),
    HiddenLayerSpec("P_Conv_12", "layer_blocks.4.2.weight", "layer_blocks.4.2.bias", 512, "conv"),
    HiddenLayerSpec("P_Dense_0", "classifier.1.weight", "classifier.1.bias", 512, "linear"),
    HiddenLayerSpec("P_Dense_1", "classifier.4.weight", "classifier.4.bias", 512, "linear"),
)

OUTPUT_LAYER_WEIGHT_KEY = "classifier.6.weight"
OUTPUT_LAYER_BIAS_KEY = "classifier.6.bias"


def validate_method_name(method: str) -> None:
    if method not in VALID_METHODS:
        raise ValueError(f"Unsupported alignment method '{method}'. Expected one of {sorted(VALID_METHODS)}.")


def clone_state_dict_to_device(
    state_dict: Mapping[str, torch.Tensor],
    device: torch.device,
) -> OrderedDict[str, torch.Tensor]:
    """Clone a state dict onto the requested device."""

    return OrderedDict((key, value.detach().clone().to(device)) for key, value in state_dict.items())


def clone_state_dict_to_cpu(
    state_dict: Mapping[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Clone a state dict onto CPU for serialization or model loading."""

    return OrderedDict((key, value.detach().cpu().clone()) for key, value in state_dict.items())


def ordered_layer_names() -> list[str]:
    return [spec.perm_name for spec in VGG16_HIDDEN_LAYER_SPECS]


def default_scale_vectors(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> OrderedDict[str, torch.Tensor]:
    """Return one-valued scales for every hidden layer."""

    return OrderedDict(
        (spec.perm_name, torch.ones(spec.size, device=device, dtype=dtype)) for spec in VGG16_HIDDEN_LAYER_SPECS
    )


def stable_sinkhorn(logits: torch.Tensor, tau: float, num_iters: int) -> torch.Tensor:
    """Compute a differentiable doubly stochastic matrix with log-space normalization."""

    if tau <= 0:
        raise ValueError(f"Sinkhorn temperature must be positive, received tau={tau}.")

    log_transport = logits / tau
    for _ in range(num_iters):
        log_transport = log_transport - torch.logsumexp(log_transport, dim=1, keepdim=True)
        log_transport = log_transport - torch.logsumexp(log_transport, dim=0, keepdim=True)
    return torch.exp(log_transport)


def permutation_matrix_from_indices(indices: np.ndarray, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Construct a permutation matrix P such that P @ x == x[indices]."""

    eye = torch.eye(len(indices), device=device, dtype=dtype)
    index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
    return eye.index_select(dim=0, index=index_tensor)


def hungarian_projection(soft_matrix: torch.Tensor) -> np.ndarray:
    """Project a soft permutation to a hard permutation via Hungarian matching."""

    row_ind, col_ind = linear_sum_assignment(soft_matrix.detach().cpu().numpy(), maximize=True)
    if not np.array_equal(row_ind, np.arange(len(row_ind))):
        raise ValueError("Unexpected Hungarian row ordering while projecting Sinkhorn matrix.")
    return col_ind.astype(np.int64)


def _matrix_with_column_scaling(base: torch.Tensor, scale_vector: torch.Tensor) -> torch.Tensor:
    """Return `base @ diag(scale_vector)` without materializing the diagonal matrix."""

    return base * scale_vector.unsqueeze(0)


class VGG16AlignmentParameters(nn.Module):
    """Trainable Sinkhorn logits and optional log-scales for VGG16 hidden layers."""

    def __init__(
        self,
        method: str,
        *,
        layer_specs: Sequence[HiddenLayerSpec] = VGG16_HIDDEN_LAYER_SPECS,
        identity_logit_strength: float = 6.0,
        logit_noise_std: float = 1e-2,
    ) -> None:
        super().__init__()
        validate_method_name(method)

        self.method = method
        self.layer_specs = tuple(layer_specs)
        self.logits = nn.ParameterDict()
        self.log_scales = nn.ParameterDict()

        for spec in self.layer_specs:
            init_logits = identity_logit_strength * torch.eye(spec.size, dtype=torch.float32)
            if logit_noise_std > 0:
                init_logits = init_logits + logit_noise_std * torch.randn_like(init_logits)
            self.logits[spec.perm_name] = nn.Parameter(init_logits)

            if self.method == METHOD_PERM_SCALE:
                self.log_scales[spec.perm_name] = nn.Parameter(torch.zeros(spec.size, dtype=torch.float32))

    def build_alignment_matrices(self, *, tau: float, sinkhorn_iters: int) -> AlignmentMatrices:
        """Build the soft monomial transport matrices for the current parameters."""

        permutations: OrderedDict[str, torch.Tensor] = OrderedDict()
        output_monomials: OrderedDict[str, torch.Tensor] = OrderedDict()
        input_transports: OrderedDict[str, torch.Tensor] = OrderedDict()
        log_scales: OrderedDict[str, torch.Tensor] = OrderedDict()
        scales: OrderedDict[str, torch.Tensor] = OrderedDict()

        for spec in self.layer_specs:
            soft_perm = stable_sinkhorn(self.logits[spec.perm_name], tau=tau, num_iters=sinkhorn_iters)
            if self.method == METHOD_PERM_SCALE:
                log_scale = self.log_scales[spec.perm_name]
                scale_vec = torch.exp(log_scale)
            else:
                log_scale = torch.zeros(spec.size, device=soft_perm.device, dtype=soft_perm.dtype)
                scale_vec = torch.ones(spec.size, device=soft_perm.device, dtype=soft_perm.dtype)

            inverse_scale_vec = torch.exp(-log_scale)
            permutations[spec.perm_name] = soft_perm
            output_monomials[spec.perm_name] = _matrix_with_column_scaling(soft_perm, scale_vec)
            input_transports[spec.perm_name] = _matrix_with_column_scaling(soft_perm, inverse_scale_vec)
            log_scales[spec.perm_name] = log_scale
            scales[spec.perm_name] = scale_vec

        return AlignmentMatrices(
            permutations=permutations,
            output_monomials=output_monomials,
            input_transports=input_transports,
            log_scales=log_scales,
            scales=scales,
        )

    def scale_regularizer(self) -> torch.Tensor:
        """Small gauge-control term used only for the scaling method."""

        if self.method != METHOD_PERM_SCALE:
            first_param = next(self.logits.parameters())
            return torch.zeros((), device=first_param.device, dtype=first_param.dtype)
        return sum(torch.sum(param * param) for param in self.log_scales.values())


def build_hard_alignment_from_soft(soft_alignment: AlignmentMatrices) -> tuple[AlignmentMatrices, OrderedDict[str, np.ndarray]]:
    """Project a soft alignment to hard permutations while retaining the learned scales."""

    hard_permutations: OrderedDict[str, np.ndarray] = OrderedDict()
    hard_monomials: OrderedDict[str, torch.Tensor] = OrderedDict()
    hard_input_transports: OrderedDict[str, torch.Tensor] = OrderedDict()
    hard_perm_matrices: OrderedDict[str, torch.Tensor] = OrderedDict()

    for layer_name, soft_perm in soft_alignment.permutations.items():
        indices = hungarian_projection(soft_perm)
        hard_perm = permutation_matrix_from_indices(indices, device=soft_perm.device, dtype=soft_perm.dtype)
        scale_vec = soft_alignment.scales[layer_name]
        inverse_scale_vec = torch.exp(-soft_alignment.log_scales[layer_name])

        hard_permutations[layer_name] = indices
        hard_perm_matrices[layer_name] = hard_perm
        hard_monomials[layer_name] = _matrix_with_column_scaling(hard_perm, scale_vec)
        hard_input_transports[layer_name] = _matrix_with_column_scaling(hard_perm, inverse_scale_vec)

    return (
        AlignmentMatrices(
            permutations=hard_perm_matrices,
            output_monomials=hard_monomials,
            input_transports=hard_input_transports,
            log_scales=_clone_tensor_dict(soft_alignment.log_scales),
            scales=_clone_tensor_dict(soft_alignment.scales),
        ),
        hard_permutations,
    )


def build_hard_alignment_from_indices(
    permutation_indices: Mapping[str, np.ndarray | Sequence[int] | torch.Tensor],
    scale_vectors: Mapping[str, torch.Tensor | np.ndarray | Sequence[float]] | None = None,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> AlignmentMatrices:
    """Construct a hard monomial alignment directly from discrete permutations and scales."""

    scales_input = scale_vectors or {}

    permutations: OrderedDict[str, torch.Tensor] = OrderedDict()
    output_monomials: OrderedDict[str, torch.Tensor] = OrderedDict()
    input_transports: OrderedDict[str, torch.Tensor] = OrderedDict()
    log_scales: OrderedDict[str, torch.Tensor] = OrderedDict()
    scales: OrderedDict[str, torch.Tensor] = OrderedDict()

    for spec in VGG16_HIDDEN_LAYER_SPECS:
        indices = permutation_indices[spec.perm_name]
        indices_np = np.asarray(indices, dtype=np.int64)
        perm_matrix = permutation_matrix_from_indices(indices_np, device=device, dtype=dtype)

        if spec.perm_name in scales_input:
            scale_vec = torch.as_tensor(scales_input[spec.perm_name], device=device, dtype=dtype)
        else:
            scale_vec = torch.ones(spec.size, device=device, dtype=dtype)
        log_scale = torch.log(scale_vec)
        inverse_scale_vec = torch.exp(-log_scale)

        permutations[spec.perm_name] = perm_matrix
        output_monomials[spec.perm_name] = _matrix_with_column_scaling(perm_matrix, scale_vec)
        input_transports[spec.perm_name] = _matrix_with_column_scaling(perm_matrix, inverse_scale_vec)
        log_scales[spec.perm_name] = log_scale
        scales[spec.perm_name] = scale_vec

    return AlignmentMatrices(
        permutations=permutations,
        output_monomials=output_monomials,
        input_transports=input_transports,
        log_scales=log_scales,
        scales=scales,
    )


def _apply_output_transform(weight: torch.Tensor, transport: torch.Tensor) -> torch.Tensor:
    flat_weight = weight.reshape(weight.shape[0], -1)
    return (transport @ flat_weight).reshape_as(weight)


def _apply_input_transport(weight: torch.Tensor, transport: torch.Tensor) -> torch.Tensor:
    reshaped = weight.reshape(weight.shape[0], transport.shape[0], -1)
    return torch.matmul(transport.unsqueeze(0), reshaped).reshape_as(weight)


def apply_alignment_to_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    alignment: AlignmentMatrices,
) -> OrderedDict[str, torch.Tensor]:
    """Transform model B into the aligned representative B'."""

    transformed = OrderedDict((key, value.clone()) for key, value in state_dict.items())
    previous_perm_name: str | None = None

    for spec in VGG16_HIDDEN_LAYER_SPECS:
        weight = state_dict[spec.weight_key]
        if previous_perm_name is not None:
            weight = _apply_input_transport(weight, alignment.input_transports[previous_perm_name])
        weight = _apply_output_transform(weight, alignment.output_monomials[spec.perm_name])

        transformed[spec.weight_key] = weight
        transformed[spec.bias_key] = alignment.output_monomials[spec.perm_name] @ state_dict[spec.bias_key]
        previous_perm_name = spec.perm_name

    if previous_perm_name is None:
        raise RuntimeError("VGG16 hidden-layer registry unexpectedly empty.")

    transformed[OUTPUT_LAYER_WEIGHT_KEY] = _apply_input_transport(
        state_dict[OUTPUT_LAYER_WEIGHT_KEY],
        alignment.input_transports[previous_perm_name],
    )
    transformed[OUTPUT_LAYER_BIAS_KEY] = state_dict[OUTPUT_LAYER_BIAS_KEY].clone()
    return transformed


def barrier_loss_on_batch(
    model: nn.Module,
    state_a: Mapping[str, torch.Tensor],
    aligned_state_b: Mapping[str, torch.Tensor],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha_grid: Sequence[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the mean interpolation loss for one batch."""

    losses = []
    for alpha in alpha_grid:
        interpolated_state = OrderedDict(
            (key, (1.0 - alpha) * state_a[key] + alpha * aligned_state_b[key]) for key in state_a
        )
        logits = functional_call(model, interpolated_state, (inputs,))
        losses.append(F.cross_entropy(logits, targets))
    loss_tensor = torch.stack(losses)
    return loss_tensor.mean(), loss_tensor


def _calibration_loader(
    *,
    data_path: str,
    calibration_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    """Build a deterministic train-subset loader with eval transforms."""

    loaders, _ = load_cifar10_eval_loaders(data_path=data_path, batch_size=batch_size, num_workers=num_workers)
    dataset = loaders["train"].dataset
    subset_size = min(calibration_size, len(dataset))
    subset = Subset(dataset, list(range(subset_size)))

    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )


def _cycle_loader(loader: DataLoader) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _serialize_matrix_dict(matrix_dict: Mapping[str, torch.Tensor]) -> Dict[str, list[list[float]]]:
    return {key: value.detach().cpu().tolist() for key, value in matrix_dict.items()}


def _serialize_vector_dict(vector_dict: Mapping[str, torch.Tensor]) -> Dict[str, list[float]]:
    return {key: value.detach().cpu().tolist() for key, value in vector_dict.items()}


def _clone_tensor_dict(tensor_dict: Mapping[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict((key, value.detach().clone()) for key, value in tensor_dict.items())


def compute_scale_statistics(scale_vectors: Mapping[str, torch.Tensor]) -> Dict[str, Any]:
    """Summarize learned positive scales for reporting and artifact logging."""

    per_layer: Dict[str, Dict[str, float]] = {}
    all_values = []
    for layer_name, scale_vector in scale_vectors.items():
        values = scale_vector.detach().cpu().to(torch.float64)
        all_values.append(values)
        per_layer[layer_name] = {
            "mean": float(values.mean().item()),
            "std": float(values.std(unbiased=False).item()),
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "l2": float(torch.linalg.vector_norm(values).item()),
        }

    if all_values:
        stacked = torch.cat(all_values)
        overall = {
            "mean": float(stacked.mean().item()),
            "std": float(stacked.std(unbiased=False).item()),
            "min": float(stacked.min().item()),
            "max": float(stacked.max().item()),
            "l2": float(torch.linalg.vector_norm(stacked).item()),
        }
    else:
        overall = {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "l2": 0.0}

    return {"overall": overall, "per_layer": per_layer}


def _training_metadata(
    *,
    method: str,
    soft_alignment: AlignmentMatrices,
    hard_permutations: Mapping[str, np.ndarray],
    history: Sequence[Dict[str, Any]],
    config_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "method": method,
        "history": list(history),
        "layer_names": ordered_layer_names(),
        "soft_permutations": _serialize_matrix_dict(soft_alignment.permutations),
        "soft_output_monomials": _serialize_matrix_dict(soft_alignment.output_monomials),
        "soft_input_transports": _serialize_matrix_dict(soft_alignment.input_transports),
        "log_scales": _serialize_vector_dict(soft_alignment.log_scales),
        "scales": _serialize_vector_dict(soft_alignment.scales),
        "hard_permutations": {key: value.tolist() for key, value in hard_permutations.items()},
        "scale_statistics": compute_scale_statistics(soft_alignment.scales),
        "config": config_snapshot,
    }


def _validate_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> None:
    expected = {spec.weight_key for spec in VGG16_HIDDEN_LAYER_SPECS}
    expected.update(spec.bias_key for spec in VGG16_HIDDEN_LAYER_SPECS)
    expected.update({OUTPUT_LAYER_WEIGHT_KEY, OUTPUT_LAYER_BIAS_KEY})
    missing = sorted(expected.difference(state_dict.keys()))
    if missing:
        raise ValueError(f"Checkpoint is missing required VGG16 parameters: {missing}")


def train_alignment_for_method(
    *,
    method: str,
    model_a_checkpoint: str,
    model_b_checkpoint: str,
    output_root: str,
    data_path: str,
    alpha_grid_train: Sequence[float],
    alignment_steps: int,
    alignment_batch_size: int,
    calibration_size: int,
    lr: float,
    tau: float,
    sinkhorn_iters: int,
    lambda_scale: float,
    device: str | torch.device,
    num_workers: int,
    seed: int,
    log_interval: int = 25,
) -> AlignmentResult:
    """Train one alignment method and persist the aligned checkpoints and artifacts."""

    validate_method_name(method)
    runtime_device = resolve_device(str(device))
    method_dir = ensure_dir(Path(output_root) / method)

    state_a_cpu = load_checkpoint_state_dict(model_a_checkpoint)
    state_b_cpu = load_checkpoint_state_dict(model_b_checkpoint)
    _validate_state_dict_keys(state_a_cpu)
    _validate_state_dict_keys(state_b_cpu)

    state_a = clone_state_dict_to_device(state_a_cpu, runtime_device)
    state_b = clone_state_dict_to_device(state_b_cpu, runtime_device)

    template_model = create_vgg16_model(num_classes=10, device=runtime_device)
    template_model.eval()
    for parameter in template_model.parameters():
        parameter.requires_grad_(False)

    train_loader = _calibration_loader(
        data_path=data_path,
        calibration_size=calibration_size,
        batch_size=alignment_batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    batch_iterator = _cycle_loader(train_loader)

    align_params = VGG16AlignmentParameters(method=method).to(runtime_device)
    optimizer = torch.optim.Adam(align_params.parameters(), lr=lr)

    history: list[Dict[str, Any]] = []
    for step in range(1, alignment_steps + 1):
        inputs, targets = next(batch_iterator)
        inputs = inputs.to(runtime_device)
        targets = targets.to(runtime_device)

        optimizer.zero_grad()
        soft_alignment = align_params.build_alignment_matrices(tau=tau, sinkhorn_iters=sinkhorn_iters)
        aligned_state_b = apply_alignment_to_state_dict(state_b, soft_alignment)

        barrier_loss, per_alpha_losses = barrier_loss_on_batch(
            template_model,
            state_a,
            aligned_state_b,
            inputs,
            targets,
            alpha_grid_train,
        )
        scale_reg = lambda_scale * align_params.scale_regularizer()
        total_loss = barrier_loss + scale_reg
        total_loss.backward()
        optimizer.step()

        step_metrics = {
            "step": step,
            "total_loss": float(total_loss.detach().cpu().item()),
            "barrier_loss": float(barrier_loss.detach().cpu().item()),
            "scale_regularizer": float(scale_reg.detach().cpu().item()),
            "alpha_losses": [float(loss.detach().cpu().item()) for loss in per_alpha_losses],
        }
        history.append(step_metrics)

        if step == 1 or step == alignment_steps or (log_interval > 0 and step % log_interval == 0):
            print(
                f"[{method}] step={step:04d} "
                f"total={step_metrics['total_loss']:.4f} "
                f"barrier={step_metrics['barrier_loss']:.4f} "
                f"scale_reg={step_metrics['scale_regularizer']:.6f}"
            )

    final_soft_alignment = align_params.build_alignment_matrices(tau=tau, sinkhorn_iters=sinkhorn_iters)
    final_soft_state = apply_alignment_to_state_dict(state_b, final_soft_alignment)
    final_hard_alignment, hard_permutations = build_hard_alignment_from_soft(final_soft_alignment)
    final_hard_state = apply_alignment_to_state_dict(state_b, final_hard_alignment)

    soft_checkpoint_path = str(method_dir / "soft_aligned.pt")
    hard_checkpoint_path = str(method_dir / "hard_aligned.pt")
    artifact_path = str(method_dir / "alignment_artifacts.pt")
    history_path = str(method_dir / "training_history.json")
    metadata_path = str(method_dir / "metadata.json")

    config_snapshot = {
        "method": method,
        "model_a_checkpoint": model_a_checkpoint,
        "model_b_checkpoint": model_b_checkpoint,
        "data_path": data_path,
        "alpha_grid_train": list(alpha_grid_train),
        "alignment_steps": alignment_steps,
        "alignment_batch_size": alignment_batch_size,
        "calibration_size": calibration_size,
        "lr": lr,
        "tau": tau,
        "sinkhorn_iters": sinkhorn_iters,
        "lambda_scale": lambda_scale,
        "device": str(runtime_device),
        "num_workers": num_workers,
        "seed": seed,
    }
    metadata = _training_metadata(
        method=method,
        soft_alignment=final_soft_alignment,
        hard_permutations=hard_permutations,
        history=history,
        config_snapshot=config_snapshot,
    )

    checkpoint_metadata = {
        "method": method,
        "artifact_path": artifact_path,
        "history_path": history_path,
        "metadata_path": metadata_path,
    }

    save_checkpoint_with_state_dict(
        model_b_checkpoint,
        soft_checkpoint_path,
        clone_state_dict_to_cpu(final_soft_state),
        metadata=checkpoint_metadata,
    )
    save_checkpoint_with_state_dict(
        model_b_checkpoint,
        hard_checkpoint_path,
        clone_state_dict_to_cpu(final_hard_state),
        metadata=checkpoint_metadata,
    )
    torch.save(metadata, artifact_path)
    write_json(history_path, {"method": method, "history": history})
    write_json(metadata_path, metadata)

    return AlignmentResult(
        method=method,
        method_dir=str(method_dir),
        soft_checkpoint_path=soft_checkpoint_path,
        hard_checkpoint_path=hard_checkpoint_path,
        artifact_path=artifact_path,
        history_path=history_path,
        metadata_path=metadata_path,
    )


def run_vgg16_alignment_experiment(
    *,
    model_a_checkpoint: str,
    model_b_checkpoint: str,
    output_root: str,
    methods: Sequence[str],
    data_path: str = "./data",
    alpha_grid_train: Sequence[float] = (0.25, 0.5, 0.75),
    alignment_steps: int = 500,
    alignment_batch_size: int = 128,
    calibration_size: int = 2048,
    lr: float = 1e-2,
    tau: float = 1.0,
    sinkhorn_iters: int = 20,
    lambda_scale: float = 1e-5,
    device: str | torch.device = "auto",
    num_workers: int = 4,
    seed: int = 0,
    log_interval: int = 25,
) -> Dict[str, AlignmentResult]:
    """Run all requested alignment methods and return their output paths."""

    root = ensure_dir(output_root)
    run_metadata = {
        "model_a_checkpoint": model_a_checkpoint,
        "model_b_checkpoint": model_b_checkpoint,
        "output_root": str(root),
        "methods": list(methods),
        "data_path": data_path,
        "alpha_grid_train": list(alpha_grid_train),
        "alignment_steps": alignment_steps,
        "alignment_batch_size": alignment_batch_size,
        "calibration_size": calibration_size,
        "lr": lr,
        "tau": tau,
        "sinkhorn_iters": sinkhorn_iters,
        "lambda_scale": lambda_scale,
        "device": str(device),
        "num_workers": num_workers,
        "seed": seed,
        "layer_names": ordered_layer_names(),
    }
    write_json(Path(root) / "run_config.json", run_metadata)

    results: Dict[str, AlignmentResult] = {}
    for method in methods:
        results[method] = train_alignment_for_method(
            method=method,
            model_a_checkpoint=model_a_checkpoint,
            model_b_checkpoint=model_b_checkpoint,
            output_root=str(root),
            data_path=data_path,
            alpha_grid_train=alpha_grid_train,
            alignment_steps=alignment_steps,
            alignment_batch_size=alignment_batch_size,
            calibration_size=calibration_size,
            lr=lr,
            tau=tau,
            sinkhorn_iters=sinkhorn_iters,
            lambda_scale=lambda_scale,
            device=device,
            num_workers=num_workers,
            seed=seed,
            log_interval=log_interval,
        )
    return results


def load_alignment_artifact(path: str) -> Dict[str, Any]:
    """Load a serialized alignment artifact."""

    artifact = torch.load(path, map_location="cpu")
    if not isinstance(artifact, dict):
        raise TypeError(f"Expected a dict artifact at {path}, received {type(artifact)!r}.")
    return artifact


def build_identity_alignment(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> AlignmentMatrices:
    """Utility for tests and no-op comparisons."""

    identity_indices = OrderedDict(
        (spec.perm_name, np.arange(spec.size, dtype=np.int64)) for spec in VGG16_HIDDEN_LAYER_SPECS
    )
    identity_scales = OrderedDict(
        (spec.perm_name, torch.ones(spec.size, device=device, dtype=dtype)) for spec in VGG16_HIDDEN_LAYER_SPECS
    )
    return build_hard_alignment_from_indices(identity_indices, identity_scales, device=device, dtype=dtype)
