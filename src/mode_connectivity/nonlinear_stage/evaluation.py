"""Validation screening, dense audit, frozen selection, and final test access."""

from __future__ import annotations

import json
from pathlib import Path

from mode_connectivity.training_stage.geometry import evaluate
from .geometry import (
    linear_profile,
    path_from_artifact,
    path_profile,
    read_endpoints,
)
from .protocol import (
    Data,
    file_hash,
    load,
    pair_by_id,
    path_dir,
    primary_pairs,
    root,
    save,
    selected_dir,
    write_json,
)


EXTRA_SPECS = [
    ("bezier", 1, 0.01),
    ("bezier", 2, 0.05),
    ("polygon", 0, 0.0),
    ("polygon", 1, 0.01),
    ("polygon", 2, 0.05),
]


def validation_path(cfg, pair_id, family, restart):
    return path_dir(cfg, pair_id, family, restart) / "validation.json"


def evaluate_validation(cfg, pair, family, restart, stop):
    artifact_path = path_dir(cfg, pair["id"], family, restart) / "path.pt"
    value = load(artifact_path)
    endpoints, _ = read_endpoints(cfg, pair)
    path = path_from_artifact(endpoints[0], endpoints[1], value).to(cfg["device"])
    data = Data(cfg, allow_test=False)
    result = dict(
        pair=pair,
        family=family,
        restart=restart,
        artifact=str(artifact_path),
        artifact_sha256=file_hash(artifact_path),
        subsets={},
    )
    for subset in ["train_eval", "validation_audit"]:
        linear = linear_profile(
            endpoints[0], endpoints[1], data.loader(subset), cfg["device"], cfg["eval_points"], stop
        )
        curve = path_profile(
            path,
            data.loader(subset),
            cfg["device"],
            cfg["eval_points"],
            linear_endpoints=dict(
                losses=[linear["losses"][0], linear["losses"][-1]],
                errors=[linear["errors"][0], linear["errors"][-1]],
            ),
            stop=stop,
        )
        result["subsets"][subset] = dict(linear=linear, curve=curve)
    write_json(validation_path(cfg, pair["id"], family, restart), result)
    return dict(pair=pair["id"], family=family, restart=restart)


def screen(cfg):
    targets, rows = [], []
    for pair in primary_pairs(cfg):
        value = json.loads(validation_path(cfg, pair["id"], "bezier", 0).read_text())
        profile = value["subsets"]["validation_audit"]["curve"]
        failed = (
            profile["loss"]["chord"] > cfg["loss_confirmation_threshold"]
            or profile["error"]["chord"] > cfg["error_confirmation_threshold"]
        )
        row = dict(
            pair=pair["id"],
            loss_chord=profile["loss"]["chord"],
            error_chord=profile["error"]["chord"],
            confirmation_required=failed,
        )
        rows.append(row)
        if failed:
            targets.append(pair["id"])
    write_json(root(cfg) / "confirmation_targets.json", dict(targets=targets, rows=rows))
    return dict(targets=len(targets), total=len(rows))


def confirmation_targets(cfg):
    path = root(cfg) / "confirmation_targets.json"
    if not path.exists():
        return []
    return json.loads(path.read_text())["targets"]


def candidate_specs(cfg, pair):
    result = [("bezier", 0)]
    positive = (
        pair["replicate"] == 0
        and pair["kind"] == "same"
        and pair["n"] == cfg["final_epoch"]
    )
    if positive or pair["id"] in confirmation_targets(cfg):
        result += [(family, restart) for family, restart, _ in EXTRA_SPECS]
    return result


def freeze_selection(cfg, pair):
    candidates = []
    for family, restart in candidate_specs(cfg, pair):
        validation = json.loads(validation_path(cfg, pair["id"], family, restart).read_text())
        profile = validation["subsets"]["validation_audit"]["curve"]
        rank = [
            profile["loss"]["chord"],
            profile["max_loss"],
            profile["loss"]["mean"],
            0 if family == "bezier" else 1,
            restart,
        ]
        candidates.append(
            dict(
                family=family,
                restart=restart,
                rank=rank,
                validation=validation,
            )
        )
    selected = min(candidates, key=lambda item: tuple(item["rank"]))
    artifact_path = Path(selected["validation"]["artifact"])
    output = dict(
        pair=pair,
        family=selected["family"],
        restart=selected["restart"],
        rank=selected["rank"],
        artifact=str(artifact_path),
        artifact_sha256=file_hash(artifact_path),
        candidates=[
            dict(family=c["family"], restart=c["restart"], rank=c["rank"])
            for c in candidates
        ],
    )
    directory = selected_dir(cfg, pair["id"])
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "selection.json", output)
    return dict(pair=pair["id"], family=output["family"], restart=output["restart"])


def evaluate_test(cfg, pair, stop):
    selection = json.loads((selected_dir(cfg, pair["id"]) / "selection.json").read_text())
    artifact_path = Path(selection["artifact"])
    if file_hash(artifact_path) != selection["artifact_sha256"]:
        raise ValueError("Selected path changed after validation freeze.")
    value = load(artifact_path)
    endpoints, _ = read_endpoints(cfg, pair)
    path = path_from_artifact(endpoints[0], endpoints[1], value).to(cfg["device"])
    data = Data(cfg, allow_test=True)
    linear = linear_profile(
        endpoints[0], endpoints[1], data.loader("test_eval"), cfg["device"], cfg["eval_points"], stop
    )
    curve = path_profile(
        path,
        data.loader("test_eval"),
        cfg["device"],
        cfg["eval_points"],
        linear_endpoints=dict(
            losses=[linear["losses"][0], linear["losses"][-1]],
            errors=[linear["errors"][0], linear["errors"][-1]],
        ),
        stop=stop,
    )
    result = dict(
        pair=pair,
        selection=selection,
        subset="test_eval",
        subset_hash=data.subsets["hashes"]["test_eval"],
        linear=linear,
        curve=curve,
    )
    write_json(selected_dir(cfg, pair["id"]) / "test.json", result)
    return dict(pair=pair["id"], points=cfg["eval_points"])


def endpoint_test(cfg, replicate):
    pair_source = cfg["source_pairs"][replicate]
    data = Data(cfg, allow_test=True)
    results = []
    for seed in pair_source["seeds"]:
        pair = dict(
            replicate=replicate,
            left_seed=seed,
            left_epoch=cfg["final_epoch"],
            right_seed=seed,
            right_epoch=cfg["final_epoch"],
        )
        endpoints, paths = read_endpoints(cfg, pair)
        metrics = evaluate(endpoints[0], data.loader("test_full"), cfg["device"])
        results.append(
            dict(seed=seed, epoch=cfg["final_epoch"], checkpoint=str(paths[0]), **metrics)
        )
    destination = root(cfg) / "endpoint_test" / f"replicate_{replicate}.json"
    write_json(destination, results)
    return dict(replicate=replicate, endpoints=len(results))


def dense_audit(cfg, stop):
    wanted = {("same", 0), ("same", 1), ("same", cfg["final_epoch"]), ("final_left", 1)}
    rows = []
    data = Data(cfg, allow_test=False)
    for pair in primary_pairs(cfg):
        if pair["replicate"] != 0 or (pair["kind"], pair["n"]) not in wanted:
            continue
        validation = json.loads(validation_path(cfg, pair["id"], "bezier", 0).read_text())
        cheap = validation["subsets"]["validation_audit"]
        endpoints, _ = read_endpoints(cfg, pair)
        value = load(path_dir(cfg, pair["id"], "bezier", 0) / "path.pt")
        path = path_from_artifact(endpoints[0], endpoints[1], value).to(cfg["device"])
        linear = linear_profile(
            endpoints[0], endpoints[1], data.loader("validation_audit"),
            cfg["device"], cfg["dense_points"], stop
        )
        curve = path_profile(
            path, data.loader("validation_audit"), cfg["device"], cfg["dense_points"],
            linear_endpoints=dict(
                losses=[linear["losses"][0], linear["losses"][-1]],
                errors=[linear["errors"][0], linear["errors"][-1]],
            ), stop=stop,
        )
        for method, dense in [("linear", linear), ("curve", curve)]:
            rows.append(
                dict(
                    pair=pair["id"],
                    method=method,
                    cheap_loss=cheap[method]["loss"]["chord"],
                    dense_loss=dense["loss"]["chord"],
                    loss_difference=dense["loss"]["chord"] - cheap[method]["loss"]["chord"],
                    cheap_error=cheap[method]["error"]["chord"],
                    dense_error=dense["error"]["chord"],
                    error_difference=dense["error"]["chord"] - cheap[method]["error"]["chord"],
                )
            )
    write_json(root(cfg) / "audit" / "dense_discrepancies.json", rows)
    return dict(comparisons=len(rows))


def pilot_gate(cfg):
    final_pair = next(
        p for p in primary_pairs(cfg)
        if p["replicate"] == 0 and p["kind"] == "same" and p["n"] == cfg["final_epoch"]
    )
    profiles = []
    for family, restart in [("bezier", 0)] + [(f, r) for f, r, _ in EXTRA_SPECS]:
        value = json.loads(validation_path(cfg, final_pair["id"], family, restart).read_text())
        profiles.append(value["subsets"]["validation_audit"]["curve"])
    if not any(
        p["loss"]["chord"] <= cfg["loss_confirmation_threshold"]
        and p["error"]["chord"] <= cfg["error_confirmation_threshold"]
        for p in profiles
    ):
        raise RuntimeError("Final-final nonlinear positive control did not pass.")
    discrepancies = json.loads((root(cfg) / "audit" / "dense_discrepancies.json").read_text())
    for row in discrepancies:
        if abs(row["loss_difference"]) > cfg["dense_loss_tolerance"]:
            raise RuntimeError(f"61-point loss profile misses a peak: {row}")
        if abs(row["error_difference"]) > cfg["dense_error_tolerance"]:
            raise RuntimeError(f"61-point error profile misses a peak: {row}")
    status_dir = root(cfg) / "status"
    checked = []
    for path in status_dir.glob("fit_*.json"):
        record = json.loads(path.read_text())
        for attempt in record.get("attempts", []):
            if attempt["host_peak_bytes_upper_bound"] >= 4 * 1024**3:
                raise RuntimeError(f"Pilot exceeded 4 GB host memory: {path.stem}")
            if attempt["seconds"] >= 2 * 60 * 60:
                raise RuntimeError(f"Pilot fit reached its two-hour allocation: {path.stem}")
        checked.append(path.stem)
    return dict(positive_control=True, dense_grid=True, checked=checked)
