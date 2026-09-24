"""Validation selection and full train/test interpolation profiles."""

from __future__ import annotations

import json

import numpy as np
import torch

from .alignment import METHODS, aligned_state, artifact_path, pair_models
from .models import device_state, interpolated_profile, read_model
from .protocol import Data, checkpoint, file_hash, pair_dir, root, verify_endpoint, write_json


def endpoint_cache_path(cfg, seed, epoch):
    return root(cfg) / "endpoint_metrics" / str(seed) / f"epoch_{epoch:03d}.json"


@torch.no_grad()
def endpoint_metrics(cfg, seed, epoch, stop):
    output = endpoint_cache_path(cfg, seed, epoch)
    if output.exists():
        return json.loads(output.read_text())
    path = checkpoint(cfg, seed, epoch)
    checkpoint_hash = verify_endpoint(cfg, path)
    model = read_model(path, cfg)
    data = Data(cfg, allow_test=True)
    result = dict(seed=seed, epoch=epoch, checkpoint_sha256=checkpoint_hash)
    for subset in ("train_report", "test_full"):
        total = wrong = count = 0
        for x, y in data.loader(subset):
            stop.check()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            logits = model(x)
            total += torch.nn.functional.cross_entropy(logits, y, reduction="sum").item()
            wrong += (logits.argmax(1) != y).sum().item()
            count += len(y)
        result[subset] = dict(loss=total / count, error=100.0 * wrong / count, count=count)
    write_json(output, result)
    return result


def evaluate_endpoint_chunk(cfg, chunk, stop):
    rows = []
    for item in chunk["items"]:
        rows.append(endpoint_metrics(cfg, int(item["seed"]), int(item["epoch"]), stop))
    return dict(endpoints=len(rows))


def _cached_endpoints(cfg, replicate, left_epoch, right_epoch, subset):
    left_seed, right_seed = cfg["seed_pairs"][replicate]
    left = json.loads(endpoint_cache_path(cfg, left_seed, left_epoch).read_text())[subset]
    right = json.loads(endpoint_cache_path(cfg, right_seed, right_epoch).read_text())[subset]
    return {"0": left, "1": right}


def validation_profiles(cfg, replicate, left_epoch, right_epoch, stop):
    destination = pair_dir(cfg, replicate, left_epoch, right_epoch) / "validation_profiles.json"
    result = json.loads(destination.read_text()) if destination.exists() else {}
    if all(method in result for method in METHODS):
        return result
    left, _, _ = pair_models(cfg, replicate, left_epoch, right_epoch)
    left_state = dict(left.named_parameters())
    loader = Data(cfg).loader("val_select")
    for method in METHODS:
        if method in result:
            continue
        stop.check()
        state, artifact = aligned_state(cfg, replicate, left_epoch, right_epoch, method)
        state = device_state(state, cfg["device"])
        profile = interpolated_profile(
            left, left_state, state, loader, cfg["selection_alphas"], stop
        )
        profile.update(
            method=method,
            subset="val_select",
            alignment_sha256=None if artifact is None else file_hash(
                artifact_path(cfg, replicate, left_epoch, right_epoch, method)
            ),
        )
        result[method] = profile
        write_json(destination, result)
        del state, artifact
    return result


def selected_methods(cfg, left_epoch, right_epoch):
    record = json.loads((root(cfg) / "selections.json").read_text())
    cell = record["selections"][f"{left_epoch:03d}_{right_epoch:03d}"]
    if "choices" in cell:
        return {
            kind: choice["method"] for kind, choice in cell["choices"].items()
        }
    overall = cell["method"]
    permutation_only = overall if overall in ("wm", "sinkhorn") else "wm"
    return {"permutation_only": permutation_only, "overall": overall}


def selected_method(cfg, left_epoch, right_epoch, kind="overall"):
    choices = selected_methods(cfg, left_epoch, right_epoch)
    return choices.get(kind, choices["overall"])


def _profile_cached_values(profile):
    return {
        f"{float(alpha):.12g}": dict(loss=loss, error=error)
        for alpha, loss, error in zip(profile["alphas"], profile["losses"], profile["errors"])
    }


def full_profiles(cfg, replicate, left_epoch, right_epoch, stop):
    destination = pair_dir(cfg, replicate, left_epoch, right_epoch) / "full_profiles.json"
    existing = json.loads(destination.read_text()) if destination.exists() else {}
    left, _, paths = pair_models(cfg, replicate, left_epoch, right_epoch)
    left_state = dict(left.named_parameters())
    choices = selected_methods(cfg, left_epoch, right_epoch)
    chosen = set(choices.values())
    data = Data(cfg, allow_test=True)
    coarse_alphas = np.linspace(0.0, 1.0, int(cfg["profile_points_all"])).tolist()
    dense_alphas = np.linspace(0.0, 1.0, int(cfg["profile_points_selected"])).tolist()
    methods = METHODS if int(replicate) == 0 else tuple(
        method for method in METHODS if method in chosen
    )
    for method in methods:
        stop.check()
        state, artifact = aligned_state(cfg, replicate, left_epoch, right_epoch, method)
        state = device_state(state, cfg["device"])
        for subset in ("train_report", "test_full"):
            coarse_key = f"{method}/{subset}/coarse"
            if int(replicate) == 0 and coarse_key not in existing:
                cached = _cached_endpoints(cfg, replicate, left_epoch, right_epoch, subset)
                existing[coarse_key] = interpolated_profile(
                    left, left_state, state, data.loader(subset), coarse_alphas, stop, cached
                )
                write_json(destination, existing)
            if method in chosen:
                dense_key = f"{method}/{subset}/selected_dense"
                if dense_key not in existing:
                    cached = (
                        _profile_cached_values(existing[coarse_key])
                        if coarse_key in existing else {}
                    )
                    cached.update(_cached_endpoints(cfg, replicate, left_epoch, right_epoch, subset))
                    existing[dense_key] = interpolated_profile(
                        left, left_state, state, data.loader(subset), dense_alphas, stop, cached
                    )
                    write_json(destination, existing)
        del state, artifact
    existing["metadata"] = dict(
        replicate=replicate,
        epochs=[left_epoch, right_epoch],
        seeds=cfg["seed_pairs"][replicate],
        endpoints=[dict(path=str(p), sha256=file_hash(p)) for p in paths],
        selected_methods=choices,
        methods=list(methods),
        all_method_profiles=int(replicate) == 0,
        full_train_examples=len(data.subsets["indices"]["train_report"]),
        full_test_examples=len(data.subsets["indices"]["test_full"]),
    )
    write_json(destination, existing)
    return existing


def evaluate_full_chunk(cfg, chunk, stop):
    for item in chunk["items"]:
        full_profiles(
            cfg, int(item["replicate"]), int(item["left_epoch"]), int(item["right_epoch"]), stop
        )
    return dict(pairs=len(chunk["items"]))
