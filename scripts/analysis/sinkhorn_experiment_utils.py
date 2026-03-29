"""Shared utilities for Sinkhorn experiment runners and exporters."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch

from scripts.analysis.run_external_sinkhorn_original_small_mnist_lmc import build_model


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


def load_model_from_checkpoint(
    model_path: Path,
    VGGClass,
    *,
    vgg_name: str,
    image_size: int,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(model_path, map_location="cpu")
    model = build_model(VGGClass, vgg_name, num_classes=10, image_size=image_size)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError(
            f"Unsupported checkpoint payload at {model_path}; expected a raw state_dict or dict with "
            "'model_state'/'state_dict'."
        )
    model.load_state_dict(normalize_state_dict_keys(state_dict))
    model.to(device)
    model.eval()
    return model


def evaluate_interpolation_results(
    model_left: torch.nn.Module,
    model_right: torch.nn.Module,
    *,
    train_loader,
    test_loader,
    lerp,
    eval_loss_acc,
    device: torch.device,
    num_eval_points: int,
) -> Dict[str, np.ndarray]:
    lambdas = np.linspace(0.0, 1.0, int(num_eval_points), dtype=np.float64)
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for lam in lambdas.tolist():
        temporal_model = lerp(model_left, model_right, float(lam))
        loss_train, acc_train = eval_loss_acc(temporal_model, train_loader, torch.nn.CrossEntropyLoss(), device)
        loss_test, acc_test = eval_loss_acc(temporal_model, test_loader, torch.nn.CrossEntropyLoss(), device)
        train_loss.append(float(loss_train))
        train_acc.append(float(acc_train) * 100.0)
        test_loss.append(float(loss_test))
        test_acc.append(float(acc_test) * 100.0)

    return {
        "ts": lambdas,
        "tr_loss": np.asarray(train_loss, dtype=np.float64),
        "tr_acc": np.asarray(train_acc, dtype=np.float64),
        "tr_err": 100.0 - np.asarray(train_acc, dtype=np.float64),
        "te_loss": np.asarray(test_loss, dtype=np.float64),
        "te_acc": np.asarray(test_acc, dtype=np.float64),
        "te_err": 100.0 - np.asarray(test_acc, dtype=np.float64),
    }


def save_interpolation_npz(output_path: Path, results: Dict[str, np.ndarray]) -> None:
    np.savez(
        output_path,
        ts=results["ts"],
        tr_loss=results["tr_loss"],
        tr_acc=results["tr_acc"],
        tr_err=results["tr_err"],
        te_loss=results["te_loss"],
        te_acc=results["te_acc"],
        te_err=results["te_err"],
        train_loss_barrier_avg=float(np.max(results["tr_loss"]) - 0.5 * (results["tr_loss"][0] + results["tr_loss"][-1])),
        test_loss_barrier_avg=float(np.max(results["te_loss"]) - 0.5 * (results["te_loss"][0] + results["te_loss"][-1])),
        train_loss_barrier_max_endpoint=float(np.max(results["tr_loss"]) - max(results["tr_loss"][0], results["tr_loss"][-1])),
        test_loss_barrier_max_endpoint=float(np.max(results["te_loss"]) - max(results["te_loss"][0], results["te_loss"][-1])),
        min_train_acc=float(np.min(results["tr_acc"])),
        min_test_acc=float(np.min(results["te_acc"])),
    )
