"""Frozen-artifact evaluation, transferred alignments, and validation audit."""

import gc
import json

import numpy as np

from .alignment import weight_artifact
from .geometry import (
    read_model,
    transformed,
    equivalence,
    profile,
    diagnostics,
    evaluate,
)
from .protocol import (
    Data,
    checkpoint,
    pair_dir,
    root,
    load,
    save,
    write_json,
    digest,
    file_hash,
    artifact_provenance,
)


METHODS = ["raw", "wm", "base", "continue", "scale", "transfer_wm", "transfer_base"]


def endpoint_metrics(cfg, net, path, data, subset):
    stat = path.stat()
    key = digest(
        [str(path), stat.st_size, stat.st_mtime_ns, data.subsets["hashes"][subset]]
    )
    cache = root(cfg) / "cache" / f"{key}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    metrics = evaluate(net, data.loader(subset), cfg["device"])
    write_json(cache, metrics)
    return metrics


def evaluate_pair(
    cfg,
    replicate,
    ea,
    eb,
    stop,
    methods=None,
    subsets=None,
    points=None,
    destination=None,
    within_seed=None,
):
    methods = METHODS if methods is None else methods
    subsets = ["train_eval", "test_eval"] if subsets is None else subsets
    directory = pair_dir(cfg, replicate, ea, eb)
    final = cfg["stages"][-1]
    final_dir = pair_dir(cfg, replicate, final, final)
    files = {
        m: (
            final_dir / f"{m[9:]}.pt"
            if m.startswith("transfer_")
            else directory / f"{m}.pt"
        )
        for m in methods
        if m != "raw"
    }
    for path in files.values():
        if not path.exists():
            raise FileNotFoundError(
                f"Alignment must be frozen before evaluation: {path}"
            )
    data = Data(cfg, allow_test=any(s.startswith("test_") for s in subsets))
    seeds = (
        cfg["seed_pairs"][replicate]
        if within_seed is None
        else [within_seed, within_seed]
    )
    pa, pb = checkpoint(cfg, seeds[0], ea), checkpoint(cfg, seeds[1], eb)
    a, b = read_model(pa, cfg), read_model(pb, cfg)
    endpoint_ids = [dict(path=str(p), sha256=file_hash(p)) for p in [pa, pb]]
    destination = destination or directory / "profiles.json"
    result = json.loads(destination.read_text()) if destination.exists() else {}
    alphas = np.linspace(0, 1, points or cfg["eval_points"]).tolist()
    endpoint_cache = {
        s: [
            endpoint_metrics(cfg, a, pa, data, s),
            endpoint_metrics(cfg, b, pb, data, s),
        ]
        for s in subsets
    }
    selection = data.loader("selection")
    for method in methods:
        stop.check()
        artifact = None if method == "raw" else load(files[method])
        aligned = b if artifact is None else transformed(b, artifact, cfg)
        difference = (
            0.0
            if artifact is None
            else equivalence(b, aligned, selection, cfg["device"], cfg)
        )
        for subset in subsets:
            key = f"{method}/{subset}"
            if key in result and result[key]["alphas"] == alphas:
                continue
            curve = profile(
                a,
                aligned,
                data.loader(subset),
                cfg["device"],
                alphas,
                endpoints=endpoint_cache[subset],
                stop=stop,
            )
            curve.update(diagnostics(a, aligned, data.loader(subset), cfg["device"]))
            curve.update(
                method=method,
                subset=subset,
                replicate=replicate,
                epochs=[ea, eb],
                seeds=seeds,
                endpoints=endpoint_ids,
                alignment_sha256=None if artifact is None else file_hash(files[method]),
                max_logit_difference=difference,
                subset_hash=data.subsets["hashes"][subset],
                scale_stats=None if artifact is None else artifact.get("scale_stats"),
            )
            result[key] = curve
            write_json(destination, result)
        if aligned is not b:
            del aligned
        del artifact
        gc.collect()
    return result


def controls(cfg, replicate, stop):
    destination = root(cfg) / "controls" / str(replicate)
    destination.mkdir(parents=True, exist_ok=True)
    seeds = cfg["seed_pairs"][replicate]
    for epoch in cfg.get("control_checkpoints", cfg["checkpoints"]):
        stop.check()
        directory = pair_dir(cfg, replicate, epoch, epoch)
        if not (directory / "wm.pt").exists():
            paths = [checkpoint(cfg, seeds[0], epoch), checkpoint(cfg, seeds[1], epoch)]
            a = read_model(paths[0], cfg)
            b = read_model(paths[1], cfg)
            artifact = weight_artifact(a, b, cfg)
            artifact.update(
                transform="weight_matching", **artifact_provenance(cfg, paths)
            )
            save(directory / "wm.pt", artifact)
            del a, b
        # Main diagonal is already evaluated by the pair jobs; reuse those profiles.
        if epoch in cfg["stages"]:
            existing = json.loads((directory / "profiles.json").read_text())
            selected = {
                k: v
                for k, v in existing.items()
                if k.split("/")[0] in ("raw", "wm", "transfer_wm", "transfer_base")
            }
            write_json(destination / f"diagonal_{epoch:03d}.json", selected)
        else:
            evaluate_pair(
                cfg,
                replicate,
                epoch,
                epoch,
                stop,
                methods=["raw", "wm", "transfer_wm", "transfer_base"],
                destination=destination / f"diagonal_{epoch:03d}.json",
            )
    for seed in seeds:
        for epoch in cfg["stages"][:-1]:
            evaluate_pair(
                cfg,
                replicate,
                epoch,
                cfg["stages"][-1],
                stop,
                methods=["raw"],
                within_seed=seed,
                destination=destination / f"within_{seed}_{epoch:03d}.json",
            )
        data = Data(cfg, allow_test=True)
        path = checkpoint(cfg, seed, cfg["epochs"])
        net = read_model(path, cfg)
        write_json(
            destination / f"endpoint_{seed}_full_test.json",
            endpoint_metrics(cfg, net, path, data, "test_full"),
        )
        del net
    return dict(replicate=replicate)


def audit(cfg, stop):
    destination = root(cfg) / "audit"
    destination.mkdir(parents=True, exist_ok=True)
    first, final = cfg["stages"][0], cfg["stages"][-1]
    differences = []
    for ea, eb in [(first, first), (first, final), (final, final)]:
        cheap = evaluate_pair(
            cfg,
            0,
            ea,
            eb,
            stop,
            methods=["raw", "base", "scale"],
            subsets=["selection"],
            destination=destination / f"{ea}_{eb}_cheap.json",
        )
        dense = evaluate_pair(
            cfg,
            0,
            ea,
            eb,
            stop,
            methods=["raw", "base", "scale"],
            subsets=["validation"],
            points=cfg["audit_points"],
            destination=destination / f"{ea}_{eb}_dense.json",
        )
        for method in ["raw", "base", "scale"]:
            for metric in ["loss", "error"]:
                differences.append(
                    dict(
                        epochs=[ea, eb],
                        method=method,
                        metric=metric,
                        cheap=cheap[f"{method}/selection"][metric]["chord"],
                        dense=dense[f"{method}/validation"][metric]["chord"],
                        difference=dense[f"{method}/validation"][metric]["chord"]
                        - cheap[f"{method}/selection"][metric]["chord"],
                    )
                )
    write_json(destination / "discrepancies.json", differences)
    return dict(comparisons=len(differences))
