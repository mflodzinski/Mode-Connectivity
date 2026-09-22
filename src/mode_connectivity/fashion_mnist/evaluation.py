"""Frozen test evaluation for one same-stage Fashion-MNIST model pair."""

from __future__ import annotations

from .alignment import aligned_states
from .nonlinear import BezierPath, path_profile
from .profiles import read_model, state_profile
from .protocol import Data, checkpoint, load, pair_dir, write_json


LINEAR_METHODS = ["raw", "permutation", "permutation_scale"]
NONLINEAR_METHODS = ["nonlinear"]
METHODS = LINEAR_METHODS + NONLINEAR_METHODS


def evaluate_linear_pair(cfg, replicate: int, epoch: int, stop):
    directory = pair_dir(cfg, replicate, epoch)
    output = directory / "linear_profiles.json"
    if output.exists():
        import json

        return json.loads(output.read_text())
    seeds = cfg["seed_pairs"][replicate]
    left = read_model(checkpoint(cfg, seeds[0], epoch), cfg)
    right = read_model(checkpoint(cfg, seeds[1], epoch), cfg)
    permuted, scaled = aligned_states(cfg, replicate, epoch)
    data = Data(cfg, allow_test=True)
    results = {}
    for subset in ["train_eval", "validation_audit", "test_full"]:
        loader = data.loader(subset)
        for method, right_state in [
            ("raw", right.state_dict()),
            ("permutation", permuted),
            ("permutation_scale", scaled),
        ]:
            results[f"{method}/{subset}"] = state_profile(
                left,
                left.state_dict(),
                right_state,
                loader,
                cfg["device"],
                int(cfg["eval_points"]),
                stop,
            )
    payload = dict(
        replicate=replicate,
        seeds=seeds,
        epoch=epoch,
        methods=LINEAR_METHODS,
        profiles=results,
    )
    write_json(output, payload)
    return payload


def evaluate_nonlinear_pair(cfg, replicate: int, epoch: int, stop):
    import json

    directory = pair_dir(cfg, replicate, epoch)
    output = directory / "nonlinear_profiles.json"
    if output.exists():
        return json.loads(output.read_text())
    seeds = cfg["seed_pairs"][replicate]
    left = read_model(checkpoint(cfg, seeds[0], epoch), cfg)
    right = read_model(checkpoint(cfg, seeds[1], epoch), cfg)
    nonlinear_artifact = load(directory / "nonlinear.pt")
    path = BezierPath(left, left.state_dict(), right.state_dict()).to(cfg["device"])
    path.restore(nonlinear_artifact["controls"])
    data = Data(cfg, allow_test=True)
    results = {}
    for subset in ["train_eval", "validation_audit", "test_full"]:
        results[f"nonlinear/{subset}"] = path_profile(
            path,
            data.loader(subset),
            cfg["device"],
            int(cfg["eval_points"]),
            stop,
        )
    payload = dict(
        replicate=replicate,
        seeds=seeds,
        epoch=epoch,
        methods=NONLINEAR_METHODS,
        profiles=results,
    )
    write_json(output, payload)
    return payload


def evaluate_pair(cfg, replicate: int, epoch: int, stop):
    """Compatibility wrapper that evaluates both independently fitted families."""

    linear = evaluate_linear_pair(cfg, replicate, epoch, stop)
    nonlinear = evaluate_nonlinear_pair(cfg, replicate, epoch, stop)
    payload = dict(
        replicate=replicate,
        seeds=linear["seeds"],
        epoch=epoch,
        methods=METHODS,
        profiles={**linear["profiles"], **nonlinear["profiles"]},
    )
    write_json(pair_dir(cfg, replicate, epoch) / "profiles.json", payload)
    return payload
