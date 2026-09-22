"""Final-endpoint WM, Sinkhorn, and Sinkhorn-plus-scale comparison."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from mode_connectivity.alignment.weight_matching import weight_matching
from mode_connectivity.external.sinkhorn_rebasin import import_external_sinkhorn

from .alignment import ScaleTransform
from .model import make_model, permutation_spec, permuted_state
from .profiles import equivalence, read_model, state_profile
from .protocol import (
    Data,
    checkpoint,
    file_hash,
    load,
    mixed_seed,
    restore_rng,
    rng_state,
    root,
    save,
    seed_all,
    write_json,
)


METHODS = ["raw", "weight_matching", "sinkhorn", "sinkhorn_scale"]


def final_alignment_root(cfg) -> Path:
    return root(cfg) / str(cfg["final_alignment_subdir"])


def final_pair_dir(cfg, replicate: int) -> Path:
    return final_alignment_root(cfg) / f"r{replicate}"


def _cpu_state(state):
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def _device_state(state, device):
    return {key: value.detach().to(device) for key, value in state.items()}


def _models(cfg, replicate: int):
    epoch = int(cfg["epochs"])
    left_seed, right_seed = cfg["seed_pairs"][replicate]
    paths = [checkpoint(cfg, left_seed, epoch), checkpoint(cfg, right_seed, epoch)]
    if any(not path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"Missing final endpoint checkpoints: {missing}")
    return (
        read_model(paths[0], cfg),
        read_model(paths[1], cfg),
        paths,
    )


def _provenance(cfg, replicate: int, paths):
    return dict(
        replicate=replicate,
        seeds=cfg["seed_pairs"][replicate],
        epoch=int(cfg["epochs"]),
        endpoints=[dict(path=str(path), sha256=file_hash(path)) for path in paths],
    )


def run_weight_matching(cfg, replicate: int, stop):
    directory = final_pair_dir(cfg, replicate)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "weight_matching.pt"
    if output.exists():
        return load(output)
    left, right, paths = _models(cfg, replicate)
    started = time.monotonic()
    with torch.no_grad():
        permutation = weight_matching(
            permutation_spec(cfg),
            left.state_dict(),
            right.state_dict(),
            max_iter=int(cfg["wm_sweeps"]),
            seed=int(cfg["alignment_seed"]),
            silent=True,
        )
        aligned = permuted_state(right.state_dict(), permutation, cfg)
    stop.check()
    selection = Data(cfg).loader("selection")
    discrepancy = equivalence(
        right,
        right.state_dict(),
        aligned,
        selection,
        cfg["device"],
        float(cfg["atol"]),
        float(cfg["rtol"]),
    )
    artifact = dict(
        permutation=permutation,
        aligned_state=_cpu_state(aligned),
        max_logit_difference=discrepancy,
        seconds=time.monotonic() - started,
        transform="weight_matching",
        **_provenance(cfg, replicate, paths),
    )
    save(output, artifact)
    return artifact


def _make_sinkhorn(cfg, right_path):
    _, RebasinNet, _ = import_external_sinkhorn()
    source = make_model(cfg)
    source.load_state_dict(load(right_path)["state_dict"])
    source.eval()
    # Upstream graph discovery runs on CPU and needs ordinary model parameters.
    for parameter in source.parameters():
        parameter.requires_grad_(True)
    module = RebasinNet(
        source,
        input_shape=(1, 1, 28, 28),
        l=float(cfg["sinkhorn_l"]),
        tau=float(cfg["sinkhorn_tau"]),
        n_iter=int(cfg["sinkhorn_iters"]),
        scale_invariant=False,
    )
    module.to(cfg["device"])
    # The vendored ``to`` method moves alignment variables but not both models.
    module.reparamnet.to(cfg["device"])
    module.identity_init()
    permutations = [parameter for parameter in module.p if parameter is not None]
    if len(permutations) != int(cfg["hidden_layers"]):
        raise RuntimeError(
            "Sinkhorn graph discovery produced "
            f"{len(permutations)} groups; expected {cfg['hidden_layers']}."
        )
    return module


def _raw_permutations(module):
    return [
        parameter.detach().cpu().clone()
        for parameter in module.p
        if parameter is not None
    ]


def _restore_permutations(module, values):
    targets = [parameter for parameter in module.p if parameter is not None]
    if len(targets) != len(values):
        raise ValueError("Sinkhorn recovery artifact has the wrong number of layers.")
    with torch.no_grad():
        for target, source in zip(targets, values):
            target.copy_(source.to(device=target.device, dtype=target.dtype))


def _hard_permutations(module):
    _, _, matching = import_external_sinkhorn()
    return [matching(value.numpy()).float() for value in _raw_permutations(module)]


def optimize_sinkhorn(cfg, replicate: int, stop):
    directory = final_pair_dir(cfg, replicate)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "sinkhorn.pt"
    if output.exists():
        return load(output)
    left, right, paths = _models(cfg, replicate)
    module = _make_sinkhorn(cfg, paths[1])
    optimizer = torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=float(cfg["sinkhorn_lr"]),
        weight_decay=float(cfg["sinkhorn_weight_decay"]),
    )
    data = Data(cfg)
    selection = data.loader("selection")
    recovery = directory / "sinkhorn_recovery.pt"
    completed = updates = stale = 0
    history, best, reference = [], None, None
    seed = int(cfg["alignment_seed"]) + 1009 * replicate + int(cfg["epochs"])
    seed_all(seed)
    if recovery.exists():
        state = load(recovery)
        _restore_permutations(module, state["raw_parameters"])
        optimizer.load_state_dict(state["optimizer"])
        completed, updates, stale = state["completed"], state["updates"], state["stale"]
        history, best, reference = state["history"], state["best"], state["reference"]
        restore_rng(state["rng"])

    def select(pass_number):
        nonlocal best, stale, reference
        module.eval()
        with torch.no_grad():
            hard_model = module()
            hard_model.eval()
            hard_state = dict(hard_model.named_parameters())
            discrepancy = equivalence(
                right,
                right.state_dict(),
                hard_state,
                selection,
                cfg["device"],
                float(cfg["atol"]),
                float(cfg["rtol"]),
            )
            profile = state_profile(
                left,
                left.state_dict(),
                hard_state,
                selection,
                cfg["device"],
                len(cfg["validation_alphas"]),
                stop,
            )
        score = [profile["loss"]["chord"], profile["loss"]["mean"], pass_number]
        if best is None or tuple(score) < tuple(best["score"]):
            best = dict(
                raw_parameters=_raw_permutations(module),
                hard_permutations=_hard_permutations(module),
                aligned_state=_cpu_state(hard_state),
                score=score,
                selection_profile=profile,
                max_logit_difference=discrepancy,
            )
        if reference is None or reference - score[0] > float(cfg["sinkhorn_min_delta"]):
            stale, reference = 0, score[0]
        else:
            stale += 1
        row = dict(pass_number=pass_number, updates=updates, score=score, stale=stale)
        history.append(row)
        save(
            recovery,
            dict(
                raw_parameters=_raw_permutations(module),
                optimizer=optimizer.state_dict(),
                completed=pass_number,
                updates=updates,
                stale=stale,
                reference=reference,
                history=history,
                best=best,
                rng=rng_state(),
            ),
        )
        write_json(directory / "sinkhorn_history.json", history)
        print(row, flush=True)

    if best is None:
        select(0)
    maximum = int(cfg["sinkhorn_passes"])
    for pass_number in range(completed + 1, maximum + 1):
        if completed >= int(cfg["sinkhorn_min_passes"]) and stale >= int(
            cfg["sinkhorn_patience"]
        ):
            break
        loader = data.loader(
            "alignment",
            batch_size=int(cfg["alignment_batch_size"]),
            order_seed=mixed_seed(seed, f"sinkhorn-pass-{pass_number}"),
        )
        module.train()
        module.reparamnet.model.eval()
        module.reparamnet.output.eval()
        for x, y in loader:
            stop.check()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            rebased = module()
            rebased.eval()
            alpha = torch.rand((), device=cfg["device"])
            params = {
                key: torch.lerp(left.state_dict()[key], value, alpha)
                for key, value in rebased.named_parameters()
            }
            logits = functional_call(left, params, (x,))
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Sinkhorn alignment loss.")
            loss.backward()
            if any(
                parameter.grad is not None and not torch.isfinite(parameter.grad).all()
                for parameter in module.parameters()
            ):
                raise FloatingPointError("Nonfinite Sinkhorn alignment gradient.")
            optimizer.step()
            updates += 1
        completed = pass_number
        if pass_number % int(cfg["validation_interval"]) == 0 or pass_number == maximum:
            select(pass_number)
    result = dict(
        **best,
        completed=completed,
        updates=updates,
        history=history,
        transform="sinkhorn",
        **_provenance(cfg, replicate, paths),
    )
    save(output, result)
    recovery.unlink(missing_ok=True)
    return result


def optimize_sinkhorn_scales(cfg, replicate: int, stop):
    directory = final_pair_dir(cfg, replicate)
    output = directory / "sinkhorn_scale.pt"
    if output.exists():
        return load(output)
    sinkhorn = optimize_sinkhorn(cfg, replicate, stop)
    left, right, paths = _models(cfg, replicate)
    base = _device_state(sinkhorn["aligned_state"], cfg["device"])
    transform = ScaleTransform(
        base, int(cfg["hidden_layers"]), int(cfg["hidden_width"])
    ).to(cfg["device"])
    optimizer = torch.optim.Adam(transform.parameters(), lr=float(cfg["scale_lr"]))
    data = Data(cfg)
    selection = data.loader("selection")
    recovery = directory / "sinkhorn_scale_recovery.pt"
    completed = updates = 0
    history, best = [], None
    seed = int(cfg["alignment_seed"]) + 2003 * replicate + int(cfg["epochs"])
    seed_all(seed)
    if recovery.exists():
        state = load(recovery)
        transform.restore(state["log_scales"])
        optimizer.load_state_dict(state["optimizer"])
        completed, updates = state["completed"], state["updates"]
        history, best = state["history"], state["best"]
        restore_rng(state["rng"])

    def select(pass_number):
        nonlocal best
        transformed = transform.transformed()
        profile = state_profile(
            left,
            left.state_dict(),
            transformed,
            selection,
            cfg["device"],
            len(cfg["validation_alphas"]),
            stop,
        )
        score = [profile["loss"]["chord"], profile["loss"]["mean"], pass_number]
        if best is None or tuple(score) < tuple(best["score"]):
            best = dict(
                log_scales=transform.snapshot(), score=score, selection_profile=profile
            )
        row = dict(pass_number=pass_number, updates=updates, score=score)
        history.append(row)
        save(
            recovery,
            dict(
                log_scales=transform.snapshot(),
                optimizer=optimizer.state_dict(),
                completed=pass_number,
                updates=updates,
                history=history,
                best=best,
                rng=rng_state(),
            ),
        )
        write_json(directory / "sinkhorn_scale_history.json", history)
        print(row, flush=True)

    if best is None:
        select(0)
    for pass_number in range(completed + 1, int(cfg["scale_passes"]) + 1):
        loader = data.loader(
            "alignment",
            batch_size=int(cfg["alignment_batch_size"]),
            order_seed=mixed_seed(seed, f"sinkhorn-scale-pass-{pass_number}"),
        )
        for x, y in loader:
            stop.check()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            alpha = torch.rand((), device=cfg["device"])
            transformed = transform.transformed()
            params = {
                key: torch.lerp(value, transformed[key], alpha)
                for key, value in left.state_dict().items()
            }
            logits = functional_call(left, params, (x,))
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss = loss + float(cfg["scale_regularization"]) * transform.regularizer()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite Sinkhorn-plus-scale loss.")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for value in transform.log_scales:
                    value.clamp_(
                        -float(cfg["scale_log_clip"]), float(cfg["scale_log_clip"])
                    )
            updates += 1
        completed = pass_number
        if pass_number % int(cfg["validation_interval"]) == 0:
            select(pass_number)
    transform.restore(best["log_scales"])
    aligned = transform.transformed()
    discrepancy = equivalence(
        right,
        right.state_dict(),
        aligned,
        selection,
        cfg["device"],
        float(cfg["atol"]),
        float(cfg["rtol"]),
    )
    result = dict(
        **best,
        aligned_state=_cpu_state(aligned),
        max_logit_difference=discrepancy,
        completed=completed,
        updates=updates,
        history=history,
        transform="sinkhorn_scale",
        source_sinkhorn_score=sinkhorn["score"],
        **_provenance(cfg, replicate, paths),
    )
    save(output, result)
    recovery.unlink(missing_ok=True)
    return result


def evaluate_final_alignment(cfg, replicate: int, stop):
    directory = final_pair_dir(cfg, replicate)
    output = directory / "profiles.json"
    if output.exists():
        return json.loads(output.read_text())
    wm = run_weight_matching(cfg, replicate, stop)
    sinkhorn = optimize_sinkhorn(cfg, replicate, stop)
    scaled = optimize_sinkhorn_scales(cfg, replicate, stop)
    left, right, _ = _models(cfg, replicate)
    states = {
        "raw": right.state_dict(),
        "weight_matching": _device_state(wm["aligned_state"], cfg["device"]),
        "sinkhorn": _device_state(sinkhorn["aligned_state"], cfg["device"]),
        "sinkhorn_scale": _device_state(scaled["aligned_state"], cfg["device"]),
    }
    data = Data(cfg, allow_test=True)
    profiles = {}
    for subset in ["train_eval", "validation_audit", "test_full"]:
        loader = data.loader(subset)
        for method, state in states.items():
            profiles[f"{method}/{subset}"] = state_profile(
                left,
                left.state_dict(),
                state,
                loader,
                cfg["device"],
                int(cfg["eval_points"]),
                stop,
            )
    payload = dict(
        replicate=replicate,
        seeds=cfg["seed_pairs"][replicate],
        epoch=int(cfg["epochs"]),
        methods=METHODS,
        profiles=profiles,
    )
    write_json(output, payload)
    return payload


def report_final_alignment(cfg):
    destination = final_alignment_root(cfg) / "report"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for replicate, seeds in enumerate(cfg["seed_pairs"]):
        payload = json.loads(
            (final_pair_dir(cfg, replicate) / "profiles.json").read_text()
        )
        for name, profile in payload["profiles"].items():
            method, subset = name.split("/")
            for metric in ["loss", "error"]:
                rows.append(
                    dict(
                        replicate=replicate,
                        left_seed=seeds[0],
                        right_seed=seeds[1],
                        method=method,
                        subset=subset,
                        metric=metric,
                        chord=profile[metric]["chord"],
                        worse=profile[metric]["worse"],
                        mean=profile[metric]["mean"],
                        peak_alpha=profile[metric]["peak_alpha"],
                    )
                )
    write_json(destination / "barriers.json", rows)
    with (destination / "barriers.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregates = []
    for subset in ["train_eval", "validation_audit", "test_full"]:
        for metric in ["loss", "error"]:
            for method in METHODS:
                values = [
                    row["chord"]
                    for row in rows
                    if row["subset"] == subset
                    and row["metric"] == metric
                    and row["method"] == method
                ]
                aggregates.append(
                    dict(
                        subset=subset,
                        metric=metric,
                        method=method,
                        n=len(values),
                        mean=float(np.mean(values)),
                        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                        values=values,
                    )
                )
    write_json(destination / "aggregates.json", aggregates)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Raw", "WM", "Sinkhorn", "Sinkhorn+scale"]
    for metric, ylabel in [
        ("loss", "Loss chord barrier"),
        ("error", "Error chord barrier (pp)"),
    ]:
        selected = [
            row
            for row in aggregates
            if row["subset"] == "test_full" and row["metric"] == metric
        ]
        selected.sort(key=lambda row: METHODS.index(row["method"]))
        means = [row["mean"] for row in selected]
        stds = [row["std"] for row in selected]
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(destination / f"final_alignment_{metric}.pdf")
        fig.savefig(destination / f"final_alignment_{metric}.png", dpi=180)
        plt.close(fig)

    comparisons = []
    for replicate, seeds in enumerate(cfg["seed_pairs"]):
        pair_rows = [
            row
            for row in rows
            if row["replicate"] == replicate
            and row["subset"] == "test_full"
            and row["metric"] == "loss"
        ]
        values = {row["method"]: row["chord"] for row in pair_rows}
        comparisons.append(
            dict(
                replicate=replicate,
                seeds=seeds,
                barriers=values,
                sinkhorn_scale_improves_sinkhorn=(
                    values["sinkhorn_scale"] < values["sinkhorn"]
                ),
                sinkhorn_beats_weight_matching=(
                    values["sinkhorn"] < values["weight_matching"]
                ),
            )
        )
    summary = dict(
        epoch=int(cfg["epochs"]),
        methods=METHODS,
        test_aggregates=[row for row in aggregates if row["subset"] == "test_full"],
        pair_comparisons=comparisons,
        figure_loss=str(destination / "final_alignment_loss.pdf"),
        figure_error=str(destination / "final_alignment_error.pdf"),
    )
    write_json(destination / "summary.json", summary)
    return summary
