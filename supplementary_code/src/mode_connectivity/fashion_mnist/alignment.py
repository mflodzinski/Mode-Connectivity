"""Hard permutation matching followed by exact positive-scale refinement."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.func import functional_call

from mode_connectivity.alignment.weight_matching import weight_matching

from .model import permutation_spec, permuted_state, positively_scaled_state
from .profiles import equivalence, read_model, state_profile
from .protocol import (
    Data,
    checkpoint,
    file_hash,
    load,
    mixed_seed,
    pair_dir,
    restore_rng,
    rng_state,
    save,
    seed_all,
    write_json,
)


def _states(cfg, replicate, epoch):
    left_seed, right_seed = cfg["seed_pairs"][replicate]
    left_path = checkpoint(cfg, left_seed, epoch)
    right_path = checkpoint(cfg, right_seed, epoch)
    left = read_model(left_path, cfg)
    right = read_model(right_path, cfg)
    return left, right, left_path, right_path


def find_permutation(cfg, replicate: int, epoch: int, stop):
    destination = pair_dir(cfg, replicate, epoch)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "permutation.pt"
    if output.exists():
        return load(output)
    left, right, left_path, right_path = _states(cfg, replicate, epoch)
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
    stop.check()
    artifact = dict(
        permutation=permutation,
        replicate=replicate,
        epoch=epoch,
        seeds=cfg["seed_pairs"][replicate],
        endpoints=[
            dict(path=str(path), sha256=file_hash(path))
            for path in [left_path, right_path]
        ],
        seconds=time.monotonic() - started,
    )
    save(output, artifact)
    return artifact


class ScaleTransform(nn.Module):
    def __init__(self, state, hidden_layers: int, width: int):
        super().__init__()
        self.hidden_layers = int(hidden_layers)
        self.state = {key: value.detach() for key, value in state.items()}
        self.log_scales = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(width, device=next(iter(state.values())).device)
                )
                for _ in range(self.hidden_layers)
            ]
        )

    def transformed(self):
        return positively_scaled_state(self.state, self.log_scales, self.hidden_layers)

    def regularizer(self):
        return torch.stack([value.square().mean() for value in self.log_scales]).mean()

    def snapshot(self):
        return [value.detach().cpu().clone() for value in self.log_scales]

    def restore(self, values):
        with torch.no_grad():
            for target, source in zip(self.log_scales, values):
                target.copy_(source.to(target.device))


def refine_scales(cfg, replicate: int, epoch: int, stop):
    directory = pair_dir(cfg, replicate, epoch)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "permutation_scale.pt"
    if output.exists():
        return load(output)
    permutation_artifact = find_permutation(cfg, replicate, epoch, stop)
    left, right, _, _ = _states(cfg, replicate, epoch)
    permuted = permuted_state(
        right.state_dict(), permutation_artifact["permutation"], cfg
    )
    transform = ScaleTransform(
        permuted, int(cfg["hidden_layers"]), int(cfg["hidden_width"])
    ).to(cfg["device"])
    optimizer = torch.optim.Adam(transform.parameters(), lr=float(cfg["scale_lr"]))
    data = Data(cfg)
    selection = data.loader("selection")
    recovery = directory / "scale_recovery.pt"
    completed, updates, history, best = 0, 0, [], None
    seed_all(int(cfg["alignment_seed"]) + 1009 * replicate + epoch)
    if recovery.exists():
        state = load(recovery)
        transform.restore(state["log_scales"])
        optimizer.load_state_dict(state["optimizer"])
        completed, updates = state["completed"], state["updates"]
        history, best = state["history"], state["best"]
        restore_rng(state["rng"])

    def select(pass_number):
        nonlocal best
        profile = state_profile(
            left,
            left.state_dict(),
            transform.transformed(),
            selection,
            cfg["device"],
            len(cfg["validation_alphas"]),
        )
        score = [profile["loss"]["chord"], profile["loss"]["mean"], pass_number]
        if best is None or tuple(score) < tuple(best["score"]):
            best = dict(log_scales=transform.snapshot(), score=score, profile=profile)
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
        write_json(directory / "scale_history.json", history)
        print(row, flush=True)

    if best is None:
        select(0)
    for pass_number in range(completed + 1, int(cfg["scale_passes"]) + 1):
        alignment = data.loader(
            "alignment",
            batch_size=int(cfg["alignment_batch_size"]),
            order_seed=mixed_seed(
                int(cfg["alignment_seed"]) + 1009 * replicate + epoch,
                f"scale-pass-{pass_number}",
            ),
        )
        for x, y in alignment:
            stop.check()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            alpha = torch.rand((), device=cfg["device"])
            right_state = transform.transformed()
            params = {
                key: torch.lerp(value, right_state[key], alpha)
                for key, value in left.state_dict().items()
            }
            logits = functional_call(left, params, (x,))
            loss = torch.nn.functional.cross_entropy(logits, y)
            loss = loss + float(cfg["scale_regularization"]) * transform.regularizer()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite scale-alignment loss.")
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
    transformed = transform.transformed()
    discrepancy = equivalence(
        right,
        right.state_dict(),
        transformed,
        selection,
        cfg["device"],
        float(cfg["atol"]),
        float(cfg["rtol"]),
    )
    result = dict(
        permutation=permutation_artifact["permutation"],
        log_scales=best["log_scales"],
        score=best["score"],
        selection_profile=best["profile"],
        max_logit_difference=discrepancy,
        updates=updates,
        completed=completed,
        replicate=replicate,
        epoch=epoch,
    )
    save(output, result)
    recovery.unlink(missing_ok=True)
    return result


def aligned_states(cfg, replicate: int, epoch: int):
    _, right, _, _ = _states(cfg, replicate, epoch)
    permutation = load(pair_dir(cfg, replicate, epoch) / "permutation.pt")
    permuted = permuted_state(right.state_dict(), permutation["permutation"], cfg)
    scale = load(pair_dir(cfg, replicate, epoch) / "permutation_scale.pt")
    scaled = positively_scaled_state(
        permuted,
        [value.to(cfg["device"]) for value in scale["log_scales"]],
        int(cfg["hidden_layers"]),
    )
    return permuted, scaled
