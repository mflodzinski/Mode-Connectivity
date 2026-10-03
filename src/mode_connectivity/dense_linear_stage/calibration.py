"""Per-cell method/hyperparameter calibration on seed pair zero only."""

from __future__ import annotations

import json
from pathlib import Path

from .alignment import (
    METHODS,
    _fit_fixed_scales,
    _fit_sinkhorn,
    artifact_aligned_state,
    artifact_path,
    fit_weight_matching,
    pair_models,
)
from .models import device_state, interpolated_profile
from .protocol import Data, load, protocol_hash, root, save, write_json


def _candidate_cfg(cfg, artifact_root: Path, method: str, candidate: dict):
    current = dict(cfg)
    current["_artifact_root"] = str(artifact_root)
    current["_ignore_frozen_hyperparameters"] = True
    current["_protocol_hash_override"] = protocol_hash(cfg)
    current["method_hyperparameters"] = {
        key: dict(value) for key, value in cfg["method_hyperparameters"].items()
    }
    values = {k: v for k, v in candidate.items() if k != "calibration_max_examples"}
    current["method_hyperparameters"][method].update(values)
    if "calibration_max_examples" in candidate:
        budget = int(candidate["calibration_max_examples"])
        current["method_hyperparameters"][method]["max_examples"] = budget
        current["method_hyperparameters"][method]["min_examples"] = min(
            int(current["method_hyperparameters"][method]["min_examples"]), budget
        )
    return current


def _main_cfg(cfg, method: str, values: dict):
    """Use a chosen candidate with the method's full scientific budget."""
    current = dict(cfg)
    current["_ignore_frozen_hyperparameters"] = True
    current["_protocol_hash_override"] = protocol_hash(cfg)
    current["method_hyperparameters"] = {
        key: dict(value) for key, value in cfg["method_hyperparameters"].items()
    }
    current["method_hyperparameters"][method].update(values)
    return current


def _cell(cfg, left_epoch, right_epoch, replicate=0):
    base = root(cfg) / "calibration"
    if cfg.get("calibrate_all_replicates", False):
        base = base / f"r{int(replicate)}"
    return base / f"{left_epoch:03d}_{right_epoch:03d}"


def _values(candidate):
    return {k: v for k, v in candidate.items() if k != "calibration_max_examples"}


def _selection_score(cfg, replicate, left_epoch, right_epoch, artifact, stop):
    """Score a fitted candidate on val_select, separate from val_tune early stopping."""
    left, _, _ = pair_models(cfg, replicate, left_epoch, right_epoch)
    state = artifact_aligned_state(
        cfg, replicate, left_epoch, right_epoch, artifact
    )
    profile = interpolated_profile(
        left,
        dict(left.named_parameters()),
        device_state(state, cfg["device"]),
        Data(cfg).loader("val_select"),
        cfg["selection_alphas"],
        stop,
    )
    loss = profile["loss"]
    return [
        float(loss["worse"]),
        float(loss["chord"]),
        float(max(profile["losses"])),
        float(loss["mean"]),
    ]


def _winner(rows, method):
    candidates = [row for row in rows if row["method"] == method]
    if not candidates:
        raise RuntimeError(f"No calibration candidates completed for {method}.")
    winner = min(candidates, key=lambda row: tuple(row["score"] + [row["candidate"]]))
    return dict(
        index=int(winner["candidate"]),
        values=_values(winner["hyperparameters"]),
        score=winner["score"],
    )


def candidate_record_path(cfg, replicate, left_epoch, right_epoch, method, index):
    return (
        _cell(cfg, left_epoch, right_epoch, replicate)
        / method
        / f"c{int(index)}"
        / "candidate.json"
    )


def fit_candidate(
    cfg, replicate, left_epoch, right_epoch, method, candidate_index, stop
):
    """Fit and independently record one hyperparameter candidate."""
    candidate_index = int(candidate_index)
    candidate = cfg["calibration_candidates"][method][candidate_index]
    candidate_root = (
        _cell(cfg, left_epoch, right_epoch, replicate)
        / method
        / f"c{candidate_index}"
    )
    current = _candidate_cfg(cfg, candidate_root, method, candidate)
    if method == "wm_scale":
        base_method = "wm"
        base = load(artifact_path(cfg, replicate, left_epoch, right_epoch, base_method))
        destination = artifact_path(
            current, replicate, left_epoch, right_epoch, base_method
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        save(destination, base)
        result = _fit_fixed_scales(
            current, replicate, left_epoch, right_epoch, method, base, stop
        )
    elif method == "sinkhorn":
        result = _fit_sinkhorn(
            current, replicate, left_epoch, right_epoch, method, stop
        )
    elif method in ("sinkhorn_scale_joint", "sinkhorn_scale_finetune"):
        base_method = "sinkhorn"
        base = load(artifact_path(cfg, replicate, left_epoch, right_epoch, base_method))
        destination = artifact_path(
            current, replicate, left_epoch, right_epoch, base_method
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        save(destination, base)
        if method == "sinkhorn_scale_joint":
            result = _fit_sinkhorn(
                current, replicate, left_epoch, right_epoch, method, stop
            )
        else:
            result = _fit_fixed_scales(
                current, replicate, left_epoch, right_epoch, method, base, stop
            )
    else:
        raise ValueError(f"Method has no candidate grid: {method}")
    row = dict(
        method=method,
        candidate=candidate_index,
        hyperparameters=candidate,
        score=_selection_score(
            cfg, replicate, left_epoch, right_epoch, result, stop
        ),
        fit_score=result["score"],
        replicate=replicate,
        epochs=[left_epoch, right_epoch],
        test_data_used=False,
    )
    write_json(
        candidate_record_path(
            cfg, replicate, left_epoch, right_epoch, method, candidate_index
        ),
        row,
    )
    return row


def fit_candidate_shard(
    cfg, replicate, left_epoch, right_epoch, method, candidate_indices, stop
):
    rows = [
        fit_candidate(
            cfg, replicate, left_epoch, right_epoch, method, index, stop
        )
        for index in candidate_indices
    ]
    return dict(
        method=method,
        candidate_indices=[int(index) for index in candidate_indices],
        rows=rows,
    )


def _recorded_candidates(cfg, replicate, left_epoch, right_epoch, method):
    rows = []
    for index in range(len(cfg["calibration_candidates"][method])):
        path = candidate_record_path(
            cfg, replicate, left_epoch, right_epoch, method, index
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing calibration candidate: {path}")
        rows.append(json.loads(path.read_text()))
    return rows


def fit_wm_cell(cfg, replicate, left_epoch, right_epoch, stop):
    result = fit_weight_matching(
        cfg, replicate, left_epoch, right_epoch, stop
    )
    return dict(
        replicate=replicate,
        epochs=[left_epoch, right_epoch],
        seconds=result.get("seconds"),
    )


def select_base_cell(cfg, replicate, left_epoch, right_epoch, stop):
    """Select parallel base-grid winners and refit them at full budget."""
    cell = _cell(cfg, left_epoch, right_epoch, replicate)
    wm = load(artifact_path(cfg, replicate, left_epoch, right_epoch, "wm"))
    rows, selected = [], {}
    for method in ("wm_scale", "sinkhorn"):
        method_rows = _recorded_candidates(
            cfg, replicate, left_epoch, right_epoch, method
        )
        rows.extend(method_rows)
        selected[method] = _winner(method_rows, method)
    _fit_fixed_scales(
        _main_cfg(cfg, "wm_scale", selected["wm_scale"]["values"]),
        replicate, left_epoch, right_epoch, "wm_scale", wm, stop,
    )
    _fit_sinkhorn(
        _main_cfg(cfg, "sinkhorn", selected["sinkhorn"]["values"]),
        replicate, left_epoch, right_epoch, "sinkhorn", stop,
    )
    record = dict(
        epochs=[left_epoch, right_epoch],
        replicate=replicate,
        candidates=rows,
        selected=selected,
        test_data_used=False,
    )
    write_json(cell / "base_choice.json", record)
    return dict(candidates=len(rows), selected=selected)


def select_branch_cell(cfg, replicate, left_epoch, right_epoch, stop):
    """Select parallel branch-grid winners, refit, and freeze method scores."""
    from .evaluation import validation_profiles

    cell = _cell(cfg, left_epoch, right_epoch, replicate)
    base_choice = json.loads((cell / "base_choice.json").read_text())
    base_artifact = load(
        artifact_path(cfg, replicate, left_epoch, right_epoch, "sinkhorn")
    )
    rows, selected = [], {}
    for method in ("sinkhorn_scale_joint", "sinkhorn_scale_finetune"):
        method_rows = _recorded_candidates(
            cfg, replicate, left_epoch, right_epoch, method
        )
        rows.extend(method_rows)
        selected[method] = _winner(method_rows, method)
    _fit_sinkhorn(
        _main_cfg(
            cfg, "sinkhorn_scale_joint",
            selected["sinkhorn_scale_joint"]["values"],
        ),
        replicate, left_epoch, right_epoch, "sinkhorn_scale_joint", stop,
    )
    _fit_fixed_scales(
        _main_cfg(
            cfg, "sinkhorn_scale_finetune",
            selected["sinkhorn_scale_finetune"]["values"],
        ),
        replicate, left_epoch, right_epoch,
        "sinkhorn_scale_finetune", base_artifact, stop,
    )
    profiles = validation_profiles(
        cfg, replicate, left_epoch, right_epoch, stop
    )
    scores = {
        method: [
            float(profile["loss"]["worse"]),
            float(profile["loss"]["chord"]),
            float(max(profile["losses"])),
            float(profile["loss"]["mean"]),
        ]
        for method, profile in profiles.items()
    }
    choices = select_method_choices(scores)
    hyperparameters = {
        "raw": {},
        "wm": {},
        "wm_scale": base_choice["selected"]["wm_scale"]["values"],
        "sinkhorn": base_choice["selected"]["sinkhorn"]["values"],
        "sinkhorn_scale_joint": selected["sinkhorn_scale_joint"]["values"],
        "sinkhorn_scale_finetune": selected["sinkhorn_scale_finetune"]["values"],
    }
    record = dict(
        epochs=[left_epoch, right_epoch],
        calibration_replicate=replicate,
        method=choices["overall"]["method"],
        validation_score=choices["overall"]["validation_score"],
        choices=choices,
        method_scores=scores,
        hyperparameters=hyperparameters,
        branch_candidates=rows,
        branch_selected=selected,
        test_data_used=False,
    )
    write_json(cell / "choice.json", record)
    return dict(choices=choices, candidates=len(rows))


def select_method_choices(scores):
    """Select strict permutation-only and unrestricted winners from one score table."""
    method_order = {method: index for index, method in enumerate(METHODS)}

    def choose(methods):
        method = min(
            methods, key=lambda value: tuple(scores[value] + [method_order[value]])
        )
        return dict(method=method, validation_score=scores[method])

    return {
        "permutation_only": choose(("wm", "sinkhorn")),
        "overall": choose(METHODS),
    }


def freeze_cell_choices(cfg):
    """Freeze per-cell choices before any official test loader is constructed."""
    if cfg.get("calibrate_all_replicates", False):
        replicate_hyperparameters, replicate_selections = {}, {}
        for replicate in range(len(cfg["seed_pairs"])):
            cells, selections = {}, {}
            for left_epoch in cfg["stages"]:
                for right_epoch in cfg["stages"]:
                    key = f"{left_epoch:03d}_{right_epoch:03d}"
                    record = json.loads(
                        (_cell(cfg, left_epoch, right_epoch, replicate) / "choice.json").read_text()
                    )
                    cells[key] = record["hyperparameters"]
                    choices = record.get("choices") or select_method_choices(record["method_scores"])
                    selections[key] = dict(
                        method=choices["overall"]["method"],
                        validation_score=choices["overall"]["validation_score"],
                        choices=choices,
                        method_scores=record["method_scores"],
                        calibration_replicate=replicate,
                    )
            replicate_hyperparameters[str(replicate)] = {"cells": cells}
            replicate_selections[str(replicate)] = selections
        write_json(root(cfg) / "hyperparameters.json", {
            "global": {},
            "replicates": replicate_hyperparameters,
            "test_data_used": False,
        })
        write_json(root(cfg) / "selections.json", dict(
            criterion=[
                "pair-specific validation-only hyperparameter selection",
                "validation loss worse",
                "validation loss chord",
                "maximum validation loss",
                "mean validation loss",
                "method order",
            ],
            methods=list(METHODS),
            stages=cfg["stages"],
            seed_pairs=cfg["seed_pairs"],
            replicates=replicate_selections,
            test_data_used=False,
        ))
        return dict(
            cells=len(cfg["stages"]) ** 2 * len(cfg["seed_pairs"]),
            calibration_replicates=len(cfg["seed_pairs"]),
        )
    cells, selections = {}, {}
    for left_epoch in cfg["stages"]:
        for right_epoch in cfg["stages"]:
            key = f"{left_epoch:03d}_{right_epoch:03d}"
            record = json.loads((_cell(cfg, left_epoch, right_epoch) / "choice.json").read_text())
            cells[key] = record["hyperparameters"]
            choices = record.get("choices")
            if choices is None:
                choices = select_method_choices(record["method_scores"])
            selections[key] = dict(
                method=choices["overall"]["method"],
                validation_score=choices["overall"]["validation_score"],
                choices=choices,
                method_scores=record["method_scores"],
                calibration_replicate=0,
            )
    write_json(root(cfg) / "hyperparameters.json", {
        "global": {},
        "cells": cells,
        "calibration_replicate": 0,
        "test_data_used": False,
    })
    write_json(root(cfg) / "selections.json", dict(
        criterion=[
            "separate permutation-only {WM, Sinkhorn} and overall selections",
            "calibration-pair validation loss worse",
            "validation loss chord",
            "maximum validation loss",
            "mean validation loss",
            "method order",
        ],
        methods=list(METHODS),
        stages=cfg["stages"],
        calibration_seed_pair=cfg["seed_pairs"][0],
        selections=selections,
        test_data_used=False,
    ))
    return dict(cells=len(cells), calibration_replicate=0)
