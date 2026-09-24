"""Full-training-data WM, Sinkhorn, and positive-scale alignment methods."""

from __future__ import annotations

import gc
import json
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.func import functional_call

from mode_connectivity.alignment.weight_matching import apply_permutation, weight_matching
from mode_connectivity.external.sinkhorn_rebasin import import_external_sinkhorn
from .models import (
    cpu_state,
    device_state,
    equivalence,
    interpolated_profile,
    make_model,
    permutation_spec,
    positive_scaled_state,
    read_model,
)
from .protocol import (
    Data,
    checkpoint,
    file_hash,
    load,
    pair_dir,
    restore_rng,
    rng_state,
    save,
    seed_all,
    write_json,
    verify_endpoint,
)


METHODS = (
    "raw",
    "wm",
    "wm_scale",
    "sinkhorn",
    "sinkhorn_scale_joint",
    "sinkhorn_scale_finetune",
)


def artifact_path(cfg, replicate, left_epoch, right_epoch, method) -> Path:
    if method in ("raw",):
        raise ValueError("Raw interpolation has no alignment artifact.")
    return pair_dir(cfg, replicate, left_epoch, right_epoch) / f"{method}.pt"


def pair_seed(cfg, replicate, left_epoch, right_epoch, offset=0) -> int:
    return (
        int(cfg["alignment_seed"])
        + 1_000_003 * int(replicate)
        + 1009 * int(left_epoch)
        + 9176 * int(right_epoch)
        + int(offset)
    ) % (2**31)


def pair_models(cfg, replicate, left_epoch, right_epoch):
    left_seed, right_seed = cfg["seed_pairs"][replicate]
    paths = [
        checkpoint(cfg, left_seed, left_epoch),
        checkpoint(cfg, right_seed, right_epoch),
    ]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError(f"Missing pair endpoint: {paths}")
    for path in paths:
        verify_endpoint(cfg, path)
    return read_model(paths[0], cfg), read_model(paths[1], cfg), paths


def provenance(cfg, replicate, left_epoch, right_epoch, paths):
    return dict(
        replicate=replicate,
        epochs=[left_epoch, right_epoch],
        seeds=cfg["seed_pairs"][replicate],
        endpoints=[dict(path=str(p), sha256=file_hash(p)) for p in paths],
    )


def artifact_aligned_state(cfg, replicate, left_epoch, right_epoch, artifact):
    """Materialize a reused permutation lazily instead of duplicating checkpoints."""
    if "aligned_state" in artifact:
        return artifact["aligned_state"]
    if artifact.get("method") == "wm" and "permutation" in artifact:
        _, right_seed = cfg["seed_pairs"][replicate]
        right_state = load(checkpoint(cfg, right_seed, right_epoch))["state_dict"]
        return cpu_state(apply_permutation(
            permutation_spec(cfg), artifact["permutation"], right_state
        ))
    raise ValueError("Alignment artifact does not contain a materializable state.")


def hyperparameters(cfg, method: str, left_epoch: int, right_epoch: int) -> dict:
    result = dict(cfg["method_hyperparameters"][method])
    path = Path(cfg["output_root"]) / "hyperparameters.json"
    if path.exists() and not cfg.get("_ignore_frozen_hyperparameters", False):
        frozen = json.loads(path.read_text())
        value = frozen.get("cells", {}).get(f"{left_epoch:03d}_{right_epoch:03d}", {}).get(method)
        value = value or frozen.get("global", {}).get(method)
        if value:
            result.update(value)
    return result


def score(profile, progress: int):
    loss = profile["loss"]
    return [
        float(loss["worse"]),
        float(loss["chord"]),
        float(max(profile["losses"])),
        float(loss["mean"]),
        int(progress),
    ]


def _better(candidate, current):
    return current is None or tuple(candidate) < tuple(current)


def _profile_state(cfg, template, left_state, right_state, loader, stop):
    return interpolated_profile(
        template,
        left_state,
        right_state,
        loader,
        cfg["validation_alphas"],
        stop,
    )


def fit_weight_matching(cfg, replicate, left_epoch, right_epoch, stop):
    output = artifact_path(cfg, replicate, left_epoch, right_epoch, "wm")
    if output.exists():
        return load(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    left, right, paths = pair_models(cfg, replicate, left_epoch, right_epoch)
    started = time.monotonic()
    with torch.no_grad():
        permutation = weight_matching(
            permutation_spec(cfg),
            left.state_dict(),
            right.state_dict(),
            max_iter=int(cfg["wm_sweeps"]),
            # Match the established WM protocol exactly so data-independent
            # artifacts from earlier runs can be reused after hash checks.
            seed=int(cfg["alignment_seed"]),
            silent=True,
        )
        aligned = apply_permutation(permutation_spec(cfg), permutation, right.state_dict())
    discrepancy = equivalence(
        right,
        right.state_dict(),
        aligned,
        Data(cfg).loader("val_tune"),
        cfg,
    )
    result = dict(
        method="wm",
        permutation=permutation,
        aligned_state=cpu_state(aligned),
        max_logit_difference=discrepancy,
        seconds=time.monotonic() - started,
        **provenance(cfg, replicate, left_epoch, right_epoch, paths),
    )
    save(output, result)
    stop.check()
    return result


class ScaleTransform(nn.Module):
    def __init__(self, state, spec, device):
        super().__init__()
        self.state = device_state(state, device)
        self.spec = spec
        sizes = {}
        for group, axes in spec.perm_to_axes.items():
            key, axis = axes[0]
            sizes[group] = self.state[key].shape[axis]
        self.log_scales = nn.ParameterDict(
            {group: nn.Parameter(torch.zeros(size, device=device)) for group, size in sizes.items()}
        )

    def transformed(self):
        return positive_scaled_state(self.state, self.log_scales, self.spec)

    def regularizer(self):
        return torch.stack([value.square().sum() for value in self.log_scales.values()]).sum()

    def snapshot(self):
        return {key: value.detach().cpu().clone() for key, value in self.log_scales.items()}

    def restore(self, values):
        with torch.no_grad():
            for key, target in self.log_scales.items():
                target.copy_(values[key].to(target.device))


def _fit_fixed_scales(
    cfg,
    replicate,
    left_epoch,
    right_epoch,
    method,
    starting_artifact,
    stop,
):
    output = artifact_path(cfg, replicate, left_epoch, right_epoch, method)
    if output.exists():
        return load(output)
    left, right, paths = pair_models(cfg, replicate, left_epoch, right_epoch)
    left_state = dict(left.named_parameters())
    transform = ScaleTransform(
        artifact_aligned_state(
            cfg, replicate, left_epoch, right_epoch, starting_artifact
        ),
        permutation_spec(cfg), cfg["device"]
    )
    hp = hyperparameters(cfg, method, left_epoch, right_epoch)
    optimizer = torch.optim.AdamW(
        transform.parameters(),
        lr=float(hp["lr"]),
        weight_decay=float(hp.get("weight_decay", 0.0)),
    )
    recovery = output.with_name(output.stem + "_recovery.pt")
    history, best, examples, updates, stale, reference, pass_number = [], None, 0, 0, 0, None, 0
    seed = pair_seed(cfg, replicate, left_epoch, right_epoch, 101 if method == "wm_scale" else 303)
    seed_all(seed)
    if recovery.exists():
        state = load(recovery)
        transform.restore(state["log_scales"])
        optimizer.load_state_dict(state["optimizer"])
        history, best = state["history"], state["best"]
        examples, updates, stale, reference, pass_number = (
            state["examples"], state["updates"], state["stale"], state["reference"], state["pass_number"]
        )
        restore_rng(state["rng"])
    data, tune = Data(cfg), Data(cfg).loader("val_tune")
    started = time.monotonic()

    def select():
        nonlocal best, stale, reference
        state = transform.transformed()
        discrepancy = equivalence(right, right.state_dict(), state, tune, cfg)
        profile = _profile_state(cfg, left, left_state, state, tune, stop)
        candidate = score(profile, examples)
        if _better(candidate, None if best is None else best["score"]):
            best = dict(
                score=candidate,
                log_scales=transform.snapshot(),
                aligned_state=cpu_state(state),
                validation_profile=profile,
                max_logit_difference=discrepancy,
            )
        if reference is None or reference - candidate[0] > float(hp["min_delta"]):
            reference, stale = candidate[0], 0
        else:
            stale += 1
        history.append(dict(pass_number=pass_number, examples=examples, updates=updates, score=candidate, stale=stale))
        save(
            recovery,
            dict(
                log_scales=transform.snapshot(), optimizer=optimizer.state_dict(), history=history,
                best=best, examples=examples, updates=updates, stale=stale, reference=reference,
                pass_number=pass_number, rng=rng_state(),
            ),
        )
        write_json(output.with_name(output.stem + "_history.json"), history)

    if best is None:
        select()
    budget = int(hp["max_examples"])
    minimum = int(hp["min_examples"])
    while examples < budget and not (examples >= minimum and stale >= int(hp["patience"])):
        pass_number += 1
        loader = data.loader(
            "train_fit", fit=True, order_seed=seed + pass_number,
            batch_size=int(cfg["alignment_batch_size"]),
        )
        for x, y in loader:
            stop.check()
            if examples >= budget:
                break
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            right_state = transform.transformed()
            alpha = torch.rand((), device=cfg["device"])
            state = {key: torch.lerp(left_state[key], right_state[key], alpha) for key in left_state}
            logits = functional_call(left, state, (x,))
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss = loss + float(hp["scale_penalty"]) * transform.regularizer()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {method} objective.")
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                for value in transform.log_scales.values():
                    value.clamp_(-float(hp["log_clip"]), float(hp["log_clip"]))
            examples += len(y)
            updates += 1
        select()
    result = dict(
        **best,
        method=method,
        examples=examples,
        updates=updates,
        passes=pass_number,
        seconds=time.monotonic() - started,
        history=history,
        hyperparameters=hp,
        source_method=starting_artifact["method"],
        **provenance(cfg, replicate, left_epoch, right_epoch, paths),
    )
    save(output, result)
    recovery.unlink(missing_ok=True)
    return result


def _make_sinkhorn(cfg, right_path, scale: bool, hp):
    _, RebasinNet, _ = import_external_sinkhorn()
    source = make_model(cfg)
    source.load_state_dict(load(right_path)["state_dict"])
    source.eval()
    for parameter in source.parameters():
        parameter.requires_grad_(True)
    input_shape = (1, 1, 28, 28) if cfg["dataset"] == "fashion_mnist" else (1, 3, 32, 32)
    module = RebasinNet(
        source,
        input_shape=input_shape,
        l=float(hp.get("sinkhorn_l", cfg["sinkhorn_l"])),
        tau=float(hp.get("tau", cfg["sinkhorn_tau"])),
        n_iter=int(hp.get("sinkhorn_iters", cfg["sinkhorn_iters"])),
        scale_invariant=scale,
        lambda_scale=float(hp.get("scale_penalty", cfg["lambda_scale"])),
    )
    module.to(cfg["device"])
    module.reparamnet.to(cfg["device"])
    module.identity_init()
    return module


def _alignment_state(module):
    return dict(
        raw_parameters=[p.detach().cpu().clone() for p in module.p if p is not None],
        raw_log_scales=[u.detach().cpu().clone() for u in getattr(module, "u", []) if u is not None],
    )


def _restore_alignment(module, state):
    with torch.no_grad():
        for attr, key in (("p", "raw_parameters"), ("u", "raw_log_scales")):
            targets = [p for p in getattr(module, attr, []) if p is not None]
            values = state.get(key, [])
            if values and len(values) != len(targets):
                raise ValueError(f"Recovery state mismatch for {key}.")
            for target, source in zip(targets, values):
                target.copy_(source.to(target.device))


def _hard_permutations(module):
    _, _, matching = import_external_sinkhorn()
    return [matching(p.detach().cpu().numpy()).float() for p in module.p if p is not None]


def _fit_sinkhorn(cfg, replicate, left_epoch, right_epoch, method, stop):
    output = artifact_path(cfg, replicate, left_epoch, right_epoch, method)
    if output.exists():
        return load(output)
    left, right, paths = pair_models(cfg, replicate, left_epoch, right_epoch)
    scale = method == "sinkhorn_scale_joint"
    hp = hyperparameters(cfg, method, left_epoch, right_epoch)
    module = _make_sinkhorn(cfg, paths[1], scale=scale, hp=hp)
    if scale:
        base = load(artifact_path(cfg, replicate, left_epoch, right_epoch, "sinkhorn"))
        _restore_alignment(module, base)
        with torch.no_grad():
            for value in module.u:
                if value is not None:
                    value.zero_()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]),
    )
    recovery = output.with_name(output.stem + "_recovery.pt")
    history, best, examples, updates, stale, reference, pass_number = [], None, 0, 0, 0, None, 0
    seed = pair_seed(cfg, replicate, left_epoch, right_epoch, 202 if not scale else 404)
    seed_all(seed)
    if recovery.exists():
        state = load(recovery)
        _restore_alignment(module, state["alignment"])
        optimizer.load_state_dict(state["optimizer"])
        history, best = state["history"], state["best"]
        examples, updates, stale, reference, pass_number = (
            state["examples"], state["updates"], state["stale"], state["reference"], state["pass_number"]
        )
        restore_rng(state["rng"])
    data, tune = Data(cfg), Data(cfg).loader("val_tune")
    left_state = dict(left.named_parameters())
    started = time.monotonic()

    def select():
        nonlocal best, stale, reference
        module.eval()
        with torch.no_grad():
            hard = module().eval()
            state = dict(hard.named_parameters())
            discrepancy = equivalence(right, right.state_dict(), state, tune, cfg)
            profile = _profile_state(cfg, left, left_state, state, tune, stop)
        candidate = score(profile, examples)
        if _better(candidate, None if best is None else best["score"]):
            snapshot = _alignment_state(module)
            best = dict(
                **snapshot,
                score=candidate,
                hard_permutations=_hard_permutations(module),
                aligned_state=cpu_state(state),
                validation_profile=profile,
                max_logit_difference=discrepancy,
            )
        if reference is None or reference - candidate[0] > float(hp["min_delta"]):
            reference, stale = candidate[0], 0
        else:
            stale += 1
        history.append(dict(pass_number=pass_number, examples=examples, updates=updates, score=candidate, stale=stale))
        save(
            recovery,
            dict(
                alignment=_alignment_state(module), optimizer=optimizer.state_dict(), history=history,
                best=best, examples=examples, updates=updates, stale=stale, reference=reference,
                pass_number=pass_number, rng=rng_state(),
            ),
        )
        write_json(output.with_name(output.stem + "_history.json"), history)

    if best is None:
        select()
    budget, minimum = int(hp["max_examples"]), int(hp["min_examples"])
    while examples < budget and not (examples >= minimum and stale >= int(hp["patience"])):
        pass_number += 1
        loader = data.loader(
            "train_fit", fit=True, order_seed=seed + pass_number,
            batch_size=int(cfg["alignment_batch_size"]),
        )
        module.train()
        module.reparamnet.model.eval()
        module.reparamnet.output.eval()
        for x, y in loader:
            stop.check()
            if examples >= budget:
                break
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            aligned = module().eval()
            alpha = torch.rand((), device=cfg["device"])
            state = {key: torch.lerp(left_state[key], value, alpha) for key, value in aligned.named_parameters()}
            logits = functional_call(left, state, (x,))
            loss = torch.nn.functional.cross_entropy(logits, y)
            if scale:
                loss = loss + module.scale_regularizer()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite {method} objective.")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in module.parameters()):
                raise FloatingPointError(f"Nonfinite {method} gradient.")
            optimizer.step()
            examples += len(y)
            updates += 1
        select()
    result = dict(
        **best,
        method=method,
        examples=examples,
        updates=updates,
        passes=pass_number,
        seconds=time.monotonic() - started,
        history=history,
        hyperparameters=hp,
        scale=scale,
        **provenance(cfg, replicate, left_epoch, right_epoch, paths),
    )
    save(output, result)
    recovery.unlink(missing_ok=True)
    return result


def fit_base_bundle(cfg, replicate, left_epoch, right_epoch, stop):
    wm = fit_weight_matching(cfg, replicate, left_epoch, right_epoch, stop)
    wm_scale = _fit_fixed_scales(
        cfg, replicate, left_epoch, right_epoch, "wm_scale", wm, stop
    )
    sinkhorn = _fit_sinkhorn(
        cfg, replicate, left_epoch, right_epoch, "sinkhorn", stop
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(
        wm_seconds=wm.get("seconds"),
        wm_scale_examples=wm_scale["examples"],
        sinkhorn_examples=sinkhorn["examples"],
    )


def fit_scale_bundle(cfg, replicate, left_epoch, right_epoch, stop):
    joint = _fit_sinkhorn(
        cfg, replicate, left_epoch, right_epoch, "sinkhorn_scale_joint", stop
    )
    base = load(artifact_path(cfg, replicate, left_epoch, right_epoch, "sinkhorn"))
    fixed = _fit_fixed_scales(
        cfg,
        replicate,
        left_epoch,
        right_epoch,
        "sinkhorn_scale_finetune",
        base,
        stop,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(joint_examples=joint["examples"], finetune_examples=fixed["examples"])


def fit_selected_methods(cfg, replicate, left_epoch, right_epoch, stop):
    """Fit the union of permutation-only and overall winners."""
    from .evaluation import selected_methods

    choices = selected_methods(cfg, left_epoch, right_epoch)
    methods = tuple(dict.fromkeys(choices.values()))
    outputs = set()

    def ensure(method):
        if method == "raw":
            return
        if method == "wm":
            fit_weight_matching(cfg, replicate, left_epoch, right_epoch, stop)
            outputs.add(artifact_path(cfg, replicate, left_epoch, right_epoch, "wm"))
            return
        if method == "wm_scale":
            wm = fit_weight_matching(cfg, replicate, left_epoch, right_epoch, stop)
            _fit_fixed_scales(
                cfg, replicate, left_epoch, right_epoch, "wm_scale", wm, stop
            )
            outputs.update({
                artifact_path(cfg, replicate, left_epoch, right_epoch, "wm"),
                artifact_path(cfg, replicate, left_epoch, right_epoch, "wm_scale"),
            })
            return
        if method == "sinkhorn":
            _fit_sinkhorn(cfg, replicate, left_epoch, right_epoch, "sinkhorn", stop)
            outputs.add(artifact_path(cfg, replicate, left_epoch, right_epoch, "sinkhorn"))
            return
        if method in ("sinkhorn_scale_joint", "sinkhorn_scale_finetune"):
            base = _fit_sinkhorn(
                cfg, replicate, left_epoch, right_epoch, "sinkhorn", stop
            )
            outputs.add(artifact_path(cfg, replicate, left_epoch, right_epoch, "sinkhorn"))
            if method == "sinkhorn_scale_joint":
                _fit_sinkhorn(cfg, replicate, left_epoch, right_epoch, method, stop)
            else:
                _fit_fixed_scales(
                    cfg, replicate, left_epoch, right_epoch, method, base, stop
                )
            outputs.add(artifact_path(cfg, replicate, left_epoch, right_epoch, method))
            return
        raise ValueError(method)

    for method in methods:
        ensure(method)
    marker = pair_dir(cfg, replicate, left_epoch, right_epoch) / "selected_fit.json"
    write_json(marker, dict(
        choices=choices,
        methods=list(methods),
        replicate=replicate,
        epochs=[left_epoch, right_epoch],
        artifacts=[str(path) for path in sorted(outputs, key=str)],
    ))
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return dict(choices=choices, methods=list(methods), artifacts=len(outputs)), [
        marker, *sorted(outputs, key=str)
    ]


def aligned_state(cfg, replicate, left_epoch, right_epoch, method):
    if method == "raw":
        _, right_seed = cfg["seed_pairs"][replicate]
        return load(checkpoint(cfg, right_seed, right_epoch))["state_dict"], None
    artifact = load(artifact_path(cfg, replicate, left_epoch, right_epoch, method))
    return artifact_aligned_state(
        cfg, replicate, left_epoch, right_epoch, artifact
    ), artifact
