"""Per-cell method/hyperparameter calibration on seed pair zero only."""

from __future__ import annotations

import json
from pathlib import Path

from .alignment import (
    METHODS,
    _fit_fixed_scales,
    _fit_sinkhorn,
    artifact_path,
    fit_weight_matching,
)
from .protocol import load, protocol_hash, root, save, write_json


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


def _cell(cfg, left_epoch, right_epoch):
    return root(cfg) / "calibration" / f"{left_epoch:03d}_{right_epoch:03d}"


def _values(candidate):
    return {k: v for k, v in candidate.items() if k != "calibration_max_examples"}


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


def _fixed_choice(cfg, method):
    values = cfg.get("fixed_hyperparameter_values", {}).get(method)
    if values is None:
        return None
    return dict(
        index="historical_global_prior",
        values=dict(values),
        score=None,
        source=cfg.get("fixed_hyperparameter_evidence", {}).get(method),
        evidence_protocol="globally frozen before this experiment",
    )


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


def calibrate_base_cell(cfg, left_epoch, right_epoch, stop):
    """Tune WM+scale and Sinkhorn, then fit their winners at full budget."""
    from .reuse import prior_for

    cell = _cell(cfg, left_epoch, right_epoch)
    wm = fit_weight_matching(cfg, 0, left_epoch, right_epoch, stop)
    rows, selected = [], {}
    for method in ("wm_scale", "sinkhorn"):
        fixed = _fixed_choice(cfg, method)
        if fixed is not None:
            selected[method] = fixed
            continue
        prior = prior_for(cfg, left_epoch, right_epoch, method)
        if prior is not None:
            selected[method] = dict(
                index="legacy_prior",
                values=prior["values"],
                score=prior["historical_score"],
                source=prior["source"],
                evidence_protocol=prior["evidence_protocol"],
            )
            continue
        for index, candidate in enumerate(cfg["calibration_candidates"][method]):
            candidate_root = cell / method / f"c{index}"
            current = _candidate_cfg(cfg, candidate_root, method, candidate)
            if method == "wm_scale":
                destination = artifact_path(current, 0, left_epoch, right_epoch, "wm")
                destination.parent.mkdir(parents=True, exist_ok=True)
                save(destination, wm)
                result = _fit_fixed_scales(
                    current, 0, left_epoch, right_epoch, method, wm, stop
                )
            else:
                result = _fit_sinkhorn(current, 0, left_epoch, right_epoch, method, stop)
            rows.append(dict(
                method=method,
                candidate=index,
                hyperparameters=candidate,
                score=result["score"],
            ))
        selected[method] = _winner(rows, method)
    _fit_fixed_scales(
        _main_cfg(cfg, "wm_scale", selected["wm_scale"]["values"]),
        0, left_epoch, right_epoch, "wm_scale", wm, stop,
    )
    _fit_sinkhorn(
        _main_cfg(cfg, "sinkhorn", selected["sinkhorn"]["values"]),
        0, left_epoch, right_epoch, "sinkhorn", stop,
    )
    record = dict(
        epochs=[left_epoch, right_epoch],
        replicate=0,
        candidates=rows,
        selected=selected,
        test_data_used=False,
    )
    write_json(cell / "base_choice.json", record)
    return dict(candidates=len(rows), selected=selected)


def calibrate_branch_cell(cfg, left_epoch, right_epoch, stop):
    """Tune both Sinkhorn scale branches and select the best of all methods."""
    from .evaluation import validation_profiles
    from .reuse import prior_for

    cell = _cell(cfg, left_epoch, right_epoch)
    base_choice = json.loads((cell / "base_choice.json").read_text())
    base_artifact = load(artifact_path(cfg, 0, left_epoch, right_epoch, "sinkhorn"))
    rows, selected = [], {}
    branch_methods = ("sinkhorn_scale_joint", "sinkhorn_scale_finetune")
    for method in branch_methods:
        fixed = _fixed_choice(cfg, method)
        if fixed is not None:
            selected[method] = fixed
            continue
        prior = prior_for(cfg, left_epoch, right_epoch, method)
        if prior is not None and method == "sinkhorn_scale_finetune":
            actual = base_choice["selected"]["sinkhorn"]["values"]
            required = prior.get("prerequisite", {})
            if any(actual.get(key) != value for key, value in required.items()):
                prior = None
        if prior is not None:
            selected[method] = dict(
                index="legacy_prior",
                values=prior["values"],
                score=prior["historical_score"],
                source=prior["source"],
                evidence_protocol=prior["evidence_protocol"],
            )
            continue
        for index, candidate in enumerate(cfg["calibration_candidates"][method]):
            candidate_root = cell / method / f"c{index}"
            current = _candidate_cfg(cfg, candidate_root, method, candidate)
            base_path = artifact_path(current, 0, left_epoch, right_epoch, "sinkhorn")
            base_path.parent.mkdir(parents=True, exist_ok=True)
            save(base_path, base_artifact)
            if method == "sinkhorn_scale_joint":
                result = _fit_sinkhorn(
                    current, 0, left_epoch, right_epoch, method, stop
                )
            else:
                result = _fit_fixed_scales(
                    current, 0, left_epoch, right_epoch, method, base_artifact, stop
                )
            rows.append(dict(
                method=method,
                candidate=index,
                hyperparameters=candidate,
                score=result["score"],
            ))
        selected[method] = _winner(rows, method)
    _fit_sinkhorn(
        _main_cfg(cfg, "sinkhorn_scale_joint", selected["sinkhorn_scale_joint"]["values"]),
        0, left_epoch, right_epoch, "sinkhorn_scale_joint", stop,
    )
    _fit_fixed_scales(
        _main_cfg(cfg, "sinkhorn_scale_finetune", selected["sinkhorn_scale_finetune"]["values"]),
        0, left_epoch, right_epoch, "sinkhorn_scale_finetune", base_artifact, stop,
    )
    profiles = validation_profiles(cfg, 0, left_epoch, right_epoch, stop)
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
        calibration_replicate=0,
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


def freeze_cell_choices(cfg):
    """Freeze per-cell choices before any official test loader is constructed."""
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
