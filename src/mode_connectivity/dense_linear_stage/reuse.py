"""Audited reuse of compatible artifacts and validation-only tuning evidence."""

from __future__ import annotations

import json
from pathlib import Path

from .alignment import artifact_path
from .protocol import checkpoint, file_hash, load, root, save, write_json


def _endpoint_hashes(cfg):
    return {
        (int(row["seed"]), int(row["epoch"])): row["sha256"]
        for row in json.loads((root(cfg) / "endpoints.json").read_text())
    }


def _artifact_endpoint_hashes(artifact):
    return [row.get("sha256") for row in artifact.get("endpoints", [])]


def _source_wm_path(source, kind, replicate, left, right):
    source = Path(source)
    if kind == "training_stage":
        return source / "pairs" / str(replicate) / f"{left:03d}_{right:03d}" / "wm.pt"
    if kind == "fashion_stage" and left == right:
        return source / "pairs" / f"r{replicate}" / f"epoch_{left:03d}" / "permutation.pt"
    return None


def _source_protocol(source):
    path = Path(source) / "protocol.json"
    if not path.exists():
        return None, None
    record = json.loads(path.read_text())
    return record.get("config", record), path


def _compatible_wm_source(cfg, spec):
    source_cfg, protocol_path = _source_protocol(spec["root"])
    if source_cfg is None:
        return False, "missing source protocol", protocol_path
    checks = {
        "wm_sweeps": int(source_cfg.get("wm_sweeps", -1)) == int(cfg["wm_sweeps"]),
        "alignment_seed": int(source_cfg.get("alignment_seed", -1)) == int(cfg["alignment_seed"]),
    }
    if cfg["dataset"] == "cifar10":
        checks["model"] = source_cfg.get("model") == cfg["model"]
    else:
        checks["hidden_layers"] = int(source_cfg.get("hidden_layers", -1)) == int(cfg["hidden_layers"])
        checks["hidden_width"] = int(source_cfg.get("hidden_width", -1)) == int(cfg["hidden_width"])
    failed = [key for key, value in checks.items() if not value]
    return not failed, ", ".join(failed), protocol_path


def _import_weight_matching(cfg, source_specs):
    expected = _endpoint_hashes(cfg)
    accepted, rejected = [], []
    for spec in source_specs:
        compatible, reason, protocol_path = _compatible_wm_source(cfg, spec)
        if not compatible:
            rejected.append(dict(source=spec["root"], reason=reason))
            continue
        for replicate, seeds in enumerate(cfg["seed_pairs"]):
            for left in cfg["stages"]:
                for right in cfg["stages"]:
                    destination = artifact_path(cfg, replicate, left, right, "wm")
                    if destination.exists():
                        continue
                    source = _source_wm_path(
                        spec["root"], spec["kind"], replicate, left, right
                    )
                    if source is None or not source.exists():
                        continue
                    artifact = load(source)
                    hashes = _artifact_endpoint_hashes(artifact)
                    wanted = [expected[(int(seeds[0]), int(left))], expected[(int(seeds[1]), int(right))]]
                    if hashes != wanted:
                        rejected.append(dict(
                            source=str(source), reason="endpoint SHA-256 mismatch",
                            observed=hashes, expected=wanted,
                        ))
                        continue
                    permutation = artifact.get("permutation", artifact.get("weight_permutation"))
                    if permutation is None:
                        rejected.append(dict(source=str(source), reason="missing permutation"))
                        continue
                    converted = dict(
                        method="wm",
                        permutation=permutation,
                        reused_lazy_state=True,
                        max_logit_difference=None,
                        seconds=0.0,
                        replicate=replicate,
                        epochs=[left, right],
                        seeds=seeds,
                        endpoints=[
                            dict(path=str(checkpoint(cfg, int(seed), int(epoch))), sha256=sha)
                            for seed, epoch, sha in zip(seeds, (left, right), wanted)
                        ],
                        reused_from=str(source.resolve()),
                        reused_sha256=file_hash(source),
                        source_protocol=str(protocol_path.resolve()),
                        source_protocol_sha256=file_hash(protocol_path),
                    )
                    save(destination, converted)
                    accepted.append(dict(
                        method="wm", replicate=replicate, epochs=[left, right],
                        source=str(source.resolve()), destination=str(destination),
                    ))
    return accepted, rejected


def _best_grid_rows(directory, phase):
    best = {}
    for path in Path(directory).glob(f"{phase}/*/grid_status/*.json"):
        record = json.loads(path.read_text())
        if record.get("status") != "complete":
            continue
        pair = record.get("pair", [])
        if len(pair) != 3 or int(pair[0]) != 0:
            continue
        score = list(record["result"]["selected_score"])
        candidate = (tuple(float(value) for value in score), record["combo"], path)
        key = (int(pair[1]), int(pair[2]))
        if key not in best or candidate[0] < best[key][0]:
            best[key] = candidate
    return best


def _hyperparameter_priors(cfg):
    result = {"cells": {}, "sources": [], "rejected": [], "test_data_used": False}
    grid = cfg.get("reuse", {}).get("vgg_alignment_grid")
    if not grid or not bool(cfg.get("reuse", {}).get("use_hyperparameter_priors", False)):
        return result
    grid = Path(grid)
    selected_base_path = grid / "selected_base.json"
    if not selected_base_path.exists():
        return result
    selected_protocol_path = grid / "selected" / "protocol.json"
    if not selected_protocol_path.exists():
        result["rejected"].append("missing selected-grid protocol")
        return result
    grid_protocol = json.loads(selected_protocol_path.read_text())
    grid_cfg = grid_protocol.get("config", grid_protocol)
    checks = {
        "model": grid_cfg.get("model") == cfg.get("model"),
        "seed_pair": grid_cfg.get("seed_pairs") == [cfg["seed_pairs"][0]],
        "alignment_seed": int(grid_cfg.get("alignment_seed", -1)) == int(cfg["alignment_seed"]),
        "sinkhorn_iters": int(grid_cfg.get("sinkhorn_iters", -1)) == int(cfg["sinkhorn_iters"]),
        "weight_decay": float(grid_cfg.get("alignment_weight_decay", -1))
        == float(cfg["method_hyperparameters"]["sinkhorn"]["weight_decay"]),
    }
    failed = [key for key, value in checks.items() if not value]
    if failed:
        result["rejected"].append("incompatible grid protocol: " + ", ".join(failed))
        return result
    selected_base = json.loads(selected_base_path.read_text())["selected"]["combo"]
    base, scale = _best_grid_rows(grid, "base"), _best_grid_rows(grid, "scale")
    for (left, right), (score, combo, path) in base.items():
        if left not in cfg["stages"] or right not in cfg["stages"]:
            continue
        key = f"{left:03d}_{right:03d}"
        result["cells"].setdefault(key, {})["sinkhorn"] = dict(
            values=dict(
                lr=float(combo["base_lr"]), tau=float(combo["tau"]),
                sinkhorn_l=float(combo["sinkhorn_l"]),
            ),
            historical_score=list(score),
            source=str(path.resolve()),
            source_sha256=file_hash(path),
            evidence_protocol="legacy 5,000-example fit and validation-only selection",
        )
        # The legacy scale sweep started from one globally selected base. Its
        # scale hyperparameters are transferable only when the cell's selected
        # base is that same prerequisite.
        if combo != selected_base or (left, right) not in scale:
            continue
        scale_score, scale_combo, scale_path = scale[(left, right)]
        result["cells"][key]["sinkhorn_scale_finetune"] = dict(
            values=dict(
                lr=float(scale_combo["scale_lr"]),
                scale_penalty=float(scale_combo["lambda_scale"]),
                weight_decay=0.01,
            ),
            historical_score=list(scale_score),
            source=str(scale_path.resolve()),
            source_sha256=file_hash(scale_path),
            prerequisite=dict(
                lr=float(selected_base["base_lr"]), tau=float(selected_base["tau"]),
                sinkhorn_l=float(selected_base["sinkhorn_l"]),
            ),
            evidence_protocol="legacy 5,000-example fit and validation-only selection",
        )
    result["sources"].append(dict(
        root=str(grid.resolve()), selected_base_sha256=file_hash(selected_base_path),
        selected_protocol_sha256=file_hash(selected_protocol_path),
    ))
    return result


def reuse_existing(cfg):
    """Import only exact WM artifacts and freeze auditable tuning priors."""
    settings = cfg.get("reuse", {})
    accepted, rejected = _import_weight_matching(cfg, settings.get("wm_sources", []))
    priors = _hyperparameter_priors(cfg)
    write_json(root(cfg) / "hyperparameter_priors.json", priors)
    record = dict(
        accepted=accepted,
        rejected=rejected,
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        prior_cells=len(priors["cells"]),
        policy=(
            "Only data-independent weight matching is imported as a final artifact. "
            "Legacy Sinkhorn/scale runs provide validation-only hyperparameter priors; "
            "their artifacts and sampled profiles are not reused."
        ),
    )
    write_json(root(cfg) / "reuse_inventory.json", record)
    return record


def prior_for(cfg, left_epoch, right_epoch, method):
    path = root(cfg) / "hyperparameter_priors.json"
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record.get("cells", {}).get(
        f"{left_epoch:03d}_{right_epoch:03d}", {}
    ).get(method)
