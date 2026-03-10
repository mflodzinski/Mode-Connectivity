"""Minimal VGG16 straight-through permutation alignment in PyTorch.

This is a lightweight PyTorch analogue of the git-rebasin STE idea:
- learn per-layer assignment logits
- build a soft Sinkhorn matrix as the surrogate
- project to a hard permutation in the forward pass
- use a straight-through estimator so gradients flow through the soft matrix

The implementation stays VGG16-specific and reuses the existing VGG16 transport
code so it remains directly comparable to the Sinkhorn baselines.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn

from scripts.lib.analysis.alignment import create_vgg16_model
from scripts.lib.alignment.permutation_pipeline import (
    load_checkpoint_state_dict,
    resolve_device,
    save_checkpoint_with_state_dict,
    write_json,
)
from scripts.lib.alignment.vgg16_sinkhorn_alignment import (
    AlignmentMatrices,
    AlignmentResult,
    VGG16_HIDDEN_LAYER_SPECS,
    _calibration_loader,
    _clone_tensor_dict,
    _cycle_loader,
    _serialize_matrix_dict,
    _serialize_vector_dict,
    _validate_state_dict_keys,
    apply_alignment_to_state_dict,
    barrier_loss_on_batch,
    build_hard_alignment_from_soft,
    clone_state_dict_to_cpu,
    clone_state_dict_to_device,
    compute_scale_statistics,
    ordered_layer_names,
    permutation_matrix_from_indices,
    stable_sinkhorn,
)
from scripts.lib.core.output import ensure_dir


METHOD_STE_PERM = "ste_perm"


def _ones_by_layer(*, device: torch.device, dtype: torch.dtype) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (spec.perm_name, torch.ones(spec.size, device=device, dtype=dtype)) for spec in VGG16_HIDDEN_LAYER_SPECS
    )


def _zeros_by_layer(*, device: torch.device, dtype: torch.dtype) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (spec.perm_name, torch.zeros(spec.size, device=device, dtype=dtype)) for spec in VGG16_HIDDEN_LAYER_SPECS
    )


def _alignment_from_permutation_matrices(
    matrices: Mapping[str, torch.Tensor],
) -> AlignmentMatrices:
    first = next(iter(matrices.values()))
    return AlignmentMatrices(
        permutations=OrderedDict((key, value) for key, value in matrices.items()),
        output_monomials=OrderedDict((key, value) for key, value in matrices.items()),
        input_transports=OrderedDict((key, value) for key, value in matrices.items()),
        log_scales=_zeros_by_layer(device=first.device, dtype=first.dtype),
        scales=_ones_by_layer(device=first.device, dtype=first.dtype),
    )


class VGG16STEParameters(nn.Module):
    """Per-layer permutation logits with a straight-through hard projection."""

    def __init__(
        self,
        *,
        identity_logit_strength: float = 6.0,
        logit_noise_std: float = 1e-2,
    ) -> None:
        super().__init__()
        self.logits = nn.ParameterDict()

        for spec in VGG16_HIDDEN_LAYER_SPECS:
            init_logits = identity_logit_strength * torch.eye(spec.size, dtype=torch.float32)
            if logit_noise_std > 0:
                init_logits = init_logits + logit_noise_std * torch.randn_like(init_logits)
            self.logits[spec.perm_name] = nn.Parameter(init_logits)

    def build_soft_alignment(self, *, tau: float, sinkhorn_iters: int) -> AlignmentMatrices:
        matrices: OrderedDict[str, torch.Tensor] = OrderedDict()
        for spec in VGG16_HIDDEN_LAYER_SPECS:
            matrices[spec.perm_name] = stable_sinkhorn(
                self.logits[spec.perm_name],
                tau=tau,
                num_iters=sinkhorn_iters,
            )
        return _alignment_from_permutation_matrices(matrices)

    def build_ste_alignment(self, *, tau: float, sinkhorn_iters: int) -> tuple[AlignmentMatrices, OrderedDict[str, torch.Tensor]]:
        soft_alignment = self.build_soft_alignment(tau=tau, sinkhorn_iters=sinkhorn_iters)
        ste_matrices: OrderedDict[str, torch.Tensor] = OrderedDict()

        for spec in VGG16_HIDDEN_LAYER_SPECS:
            soft_perm = soft_alignment.permutations[spec.perm_name]
            hard_alignment, _ = build_hard_alignment_from_soft(
                _alignment_from_permutation_matrices(OrderedDict([(spec.perm_name, soft_perm)]))
            )
            hard_perm = hard_alignment.permutations[spec.perm_name]
            ste_perm = soft_perm + (hard_perm - soft_perm).detach()
            ste_matrices[spec.perm_name] = ste_perm

        return _alignment_from_permutation_matrices(ste_matrices), soft_alignment.permutations


def _training_metadata(
    *,
    soft_alignment: AlignmentMatrices,
    hard_permutations: Mapping[str, Sequence[int]],
    history: Sequence[Dict[str, Any]],
    config_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "method": METHOD_STE_PERM,
        "history": list(history),
        "layer_names": ordered_layer_names(),
        "soft_permutations": _serialize_matrix_dict(soft_alignment.permutations),
        "soft_output_monomials": _serialize_matrix_dict(soft_alignment.output_monomials),
        "soft_input_transports": _serialize_matrix_dict(soft_alignment.input_transports),
        "log_scales": _serialize_vector_dict(soft_alignment.log_scales),
        "scales": _serialize_vector_dict(soft_alignment.scales),
        "hard_permutations": {key: list(map(int, value)) for key, value in hard_permutations.items()},
        "scale_statistics": compute_scale_statistics(soft_alignment.scales),
        "config": config_snapshot,
    }


def train_vgg16_ste_alignment(
    *,
    dataset: str,
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
    device: str | torch.device,
    num_workers: int,
    seed: int,
    log_interval: int = 25,
) -> AlignmentResult:
    runtime_device = resolve_device(str(device))
    method_dir = ensure_dir(Path(output_root) / METHOD_STE_PERM)

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
        dataset=dataset,
        data_path=data_path,
        calibration_size=calibration_size,
        batch_size=alignment_batch_size,
        num_workers=num_workers,
        seed=seed,
    )
    batch_iterator = _cycle_loader(train_loader)

    align_params = VGG16STEParameters().to(runtime_device)
    optimizer = torch.optim.Adam(align_params.parameters(), lr=lr)

    history: list[Dict[str, Any]] = []
    for step in range(1, alignment_steps + 1):
        inputs, targets = next(batch_iterator)
        inputs = inputs.to(runtime_device)
        targets = targets.to(runtime_device)

        optimizer.zero_grad()
        ste_alignment, soft_matrices = align_params.build_ste_alignment(tau=tau, sinkhorn_iters=sinkhorn_iters)
        aligned_state_b = apply_alignment_to_state_dict(state_b, ste_alignment)
        barrier_loss, per_alpha_losses = barrier_loss_on_batch(
            template_model,
            state_a,
            aligned_state_b,
            inputs,
            targets,
            alpha_grid_train,
        )
        barrier_loss.backward()
        optimizer.step()

        soft_sharpness = float(
            torch.stack([matrix.max(dim=1).values.mean() for matrix in soft_matrices.values()]).mean().detach().cpu().item()
        )
        step_metrics = {
            "step": step,
            "total_loss": float(barrier_loss.detach().cpu().item()),
            "barrier_loss": float(barrier_loss.detach().cpu().item()),
            "alpha_losses": [float(loss.detach().cpu().item()) for loss in per_alpha_losses],
            "mean_row_max_soft_perm": soft_sharpness,
        }
        history.append(step_metrics)

        if step == 1 or step == alignment_steps or (log_interval > 0 and step % log_interval == 0):
            print(
                f"[{METHOD_STE_PERM}] step={step:04d} "
                f"loss={step_metrics['total_loss']:.4f} "
                f"mean_row_max={step_metrics['mean_row_max_soft_perm']:.4f}"
            )

    final_soft_alignment = align_params.build_soft_alignment(tau=tau, sinkhorn_iters=sinkhorn_iters)
    final_soft_state = apply_alignment_to_state_dict(state_b, final_soft_alignment)
    final_hard_alignment, hard_permutations = build_hard_alignment_from_soft(final_soft_alignment)
    final_hard_state = apply_alignment_to_state_dict(state_b, final_hard_alignment)

    soft_checkpoint_path = str(method_dir / "soft_aligned.pt")
    hard_checkpoint_path = str(method_dir / "hard_aligned.pt")
    artifact_path = str(method_dir / "alignment_artifacts.pt")
    history_path = str(method_dir / "training_history.json")
    metadata_path = str(method_dir / "metadata.json")

    config_snapshot = {
        "method": METHOD_STE_PERM,
        "dataset": dataset,
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
        "device": str(runtime_device),
        "num_workers": num_workers,
        "seed": seed,
    }
    metadata = _training_metadata(
        soft_alignment=final_soft_alignment,
        hard_permutations=hard_permutations,
        history=history,
        config_snapshot=config_snapshot,
    )

    checkpoint_metadata = {
        "method": METHOD_STE_PERM,
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
    write_json(history_path, {"method": METHOD_STE_PERM, "history": history})
    write_json(metadata_path, metadata)

    return AlignmentResult(
        method=METHOD_STE_PERM,
        method_dir=str(method_dir),
        soft_checkpoint_path=soft_checkpoint_path,
        hard_checkpoint_path=hard_checkpoint_path,
        artifact_path=artifact_path,
        history_path=history_path,
        metadata_path=metadata_path,
    )


def run_vgg16_ste_alignment_experiment(
    *,
    model_a_checkpoint: str,
    model_b_checkpoint: str,
    output_root: str,
    dataset: str = "MNIST",
    data_path: str = "./data",
    alpha_grid_train: Sequence[float] = (0.25, 0.5, 0.75),
    alignment_steps: int = 500,
    alignment_batch_size: int = 128,
    calibration_size: int = 2048,
    lr: float = 1e-2,
    tau: float = 1.0,
    sinkhorn_iters: int = 20,
    device: str | torch.device = "auto",
    num_workers: int = 4,
    seed: int = 0,
    log_interval: int = 25,
) -> AlignmentResult:
    root = ensure_dir(output_root)
    run_metadata = {
        "model_a_checkpoint": model_a_checkpoint,
        "model_b_checkpoint": model_b_checkpoint,
        "output_root": str(root),
        "method": METHOD_STE_PERM,
        "dataset": dataset,
        "data_path": data_path,
        "alpha_grid_train": list(alpha_grid_train),
        "alignment_steps": alignment_steps,
        "alignment_batch_size": alignment_batch_size,
        "calibration_size": calibration_size,
        "lr": lr,
        "tau": tau,
        "sinkhorn_iters": sinkhorn_iters,
        "device": str(device),
        "num_workers": num_workers,
        "seed": seed,
        "layer_names": ordered_layer_names(),
    }
    write_json(Path(root) / "run_config.json", run_metadata)

    return train_vgg16_ste_alignment(
        dataset=dataset,
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
        device=device,
        num_workers=num_workers,
        seed=seed,
        log_interval=log_interval,
    )
