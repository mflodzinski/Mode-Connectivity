"""Raw-endpoint quadratic Bézier path fitting."""

from __future__ import annotations

import time

import torch
from torch import nn
from torch.func import functional_call

from .profiles import read_model
from .protocol import (
    Data,
    checkpoint,
    load,
    pair_dir,
    restore_rng,
    rng_state,
    save,
    seed_all,
    write_json,
)


class BezierPath(nn.Module):
    def __init__(self, template, left, right):
        super().__init__()
        self.template = template
        self.names = list(left)
        self.left = {key: value.detach() for key, value in left.items()}
        self.right = {key: value.detach() for key, value in right.items()}
        self.controls = nn.ParameterList(
            [
                nn.Parameter(torch.lerp(self.left[key], self.right[key], 0.5))
                for key in self.names
            ]
        )

    def state_at(self, alpha):
        alpha = torch.as_tensor(
            alpha,
            device=self.controls[0].device,
            dtype=self.controls[0].dtype,
        )
        return {
            key: (1 - alpha).square() * self.left[key]
            + 2 * alpha * (1 - alpha) * control
            + alpha.square() * self.right[key]
            for key, control in zip(self.names, self.controls)
        }

    def forward(self, x, alpha):
        return functional_call(self.template, self.state_at(alpha), (x,))

    def snapshot(self):
        return [value.detach().cpu().clone() for value in self.controls]

    def restore(self, values):
        with torch.no_grad():
            for target, source in zip(self.controls, values):
                target.copy_(source.to(target.device))


@torch.no_grad()
def path_profile(path, loader, device, points, stop=None):
    alphas = torch.linspace(0, 1, int(points)).tolist()
    losses, errors = [], []
    for alpha in alphas:
        if stop is not None:
            stop.check()
        total, wrong, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = path(x, alpha)
            total += torch.nn.functional.cross_entropy(
                logits, y, reduction="sum"
            ).item()
            wrong += (logits.argmax(1) != y).sum().item()
            count += len(y)
        losses.append(total / count)
        errors.append(100.0 * wrong / count)
    from mode_connectivity.training_stage.geometry import barriers

    return dict(
        alphas=alphas,
        losses=losses,
        errors=errors,
        loss=barriers(alphas, losses),
        error=barriers(alphas, errors),
    )


def fit_nonlinear(cfg, replicate: int, epoch: int, stop):
    directory = pair_dir(cfg, replicate, epoch)
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "nonlinear.pt"
    if output.exists():
        return load(output)
    seeds = cfg["seed_pairs"][replicate]
    left = read_model(checkpoint(cfg, seeds[0], epoch), cfg)
    right = read_model(checkpoint(cfg, seeds[1], epoch), cfg)
    path = BezierPath(left, left.state_dict(), right.state_dict()).to(cfg["device"])
    optimizer = torch.optim.SGD(
        path.controls,
        lr=float(cfg["nonlinear_lr"]),
        momentum=float(cfg["nonlinear_momentum"]),
    )
    data = Data(cfg)
    selection = data.loader("selection")
    seed = int(cfg["path_seed"]) + 1009 * replicate + epoch
    seed_all(seed)
    recovery = directory / "nonlinear_recovery.pt"
    completed, updates, history, best = 0, 0, [], None
    if recovery.exists():
        state = load(recovery)
        path.restore(state["controls"])
        optimizer.load_state_dict(state["optimizer"])
        completed, updates = state["completed"], state["updates"]
        history, best = state["history"], state["best"]
        restore_rng(state["rng"])
    started = time.monotonic()

    def select(pass_number):
        nonlocal best
        profile = path_profile(
            path, selection, cfg["device"], len(cfg["validation_alphas"])
        )
        score = [profile["loss"]["chord"], profile["loss"]["mean"], pass_number]
        if best is None or tuple(score) < tuple(best["score"]):
            best = dict(controls=path.snapshot(), score=score, profile=profile)
        row = dict(
            pass_number=pass_number,
            updates=updates,
            score=score,
            seconds=time.monotonic() - started,
        )
        history.append(row)
        save(
            recovery,
            dict(
                controls=path.snapshot(),
                optimizer=optimizer.state_dict(),
                completed=pass_number,
                updates=updates,
                history=history,
                best=best,
                rng=rng_state(),
            ),
        )
        write_json(directory / "nonlinear_history.json", history)
        print(row, flush=True)

    if best is None:
        select(0)
    for pass_number in range(completed + 1, int(cfg["nonlinear_passes"]) + 1):
        loader = data.loader(
            "curve_fit",
            batch_size=int(cfg["nonlinear_batch_size"]),
            order_seed=seed + pass_number,
        )
        for x, y in loader:
            stop.check()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            alpha = torch.rand((), device=cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = path(x, alpha)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite nonlinear-path loss.")
            loss.backward()
            optimizer.step()
            updates += 1
        completed = pass_number
        if pass_number % int(cfg["validation_interval"]) == 0:
            select(pass_number)
    result = dict(
        controls=best["controls"],
        score=best["score"],
        selection_profile=best["profile"],
        updates=updates,
        completed=completed,
        replicate=replicate,
        epoch=epoch,
        family="quadratic_bezier",
    )
    save(output, result)
    recovery.unlink(missing_ok=True)
    return result
