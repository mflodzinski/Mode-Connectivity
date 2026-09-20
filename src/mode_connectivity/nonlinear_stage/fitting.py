"""Resumable Garipov-style fitting for one frozen endpoint pair."""

from __future__ import annotations

import time

import torch

from .geometry import PathModel, artifact, path_geometry, path_profile, read_endpoints
from .protocol import (
    Data,
    StopRequested,
    load,
    path_dir,
    restore_rng,
    rng_state,
    save,
    seed_all,
    seed_for,
    write_json,
)


def learning_rate(base, completed_passes, total_passes):
    alpha = completed_passes / total_passes
    if alpha <= 0.5:
        factor = 1.0
    elif alpha <= 0.9:
        factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
    else:
        factor = 0.01
    return base * factor


def score(profile, completed_passes):
    return (
        float(profile["loss"]["chord"]),
        float(profile["max_loss"]),
        float(profile["loss"]["mean"]),
        int(completed_passes),
    )


def fit(cfg, pair, family, restart, noise, stop):
    directory = path_dir(cfg, pair["id"], family, restart)
    directory.mkdir(parents=True, exist_ok=True)
    recovery_path = directory / "recovery.pt"
    history_path, output_path = directory / "history.json", directory / "path.pt"
    seed = seed_for(cfg, pair["id"], family, restart)
    seed_all(seed)
    endpoints, _ = read_endpoints(cfg, pair)
    generator = torch.Generator(device=cfg["device"]).manual_seed(seed)
    path = PathModel(endpoints[0], endpoints[1], family, noise, generator).to(cfg["device"])
    optimizer = torch.optim.SGD(
        path.controls.parameters(), lr=cfg["fit_lr"], momentum=cfg["fit_momentum"]
    )
    data = Data(cfg, allow_test=False)
    selection = data.loader("selection")
    completed, updates, examples, history = 0, 0, 0, []
    if recovery_path.exists():
        state = load(recovery_path)
        path.load_control_state(state["controls"])
        optimizer.load_state_dict(state["optimizer"])
        completed, updates, examples = state["pass"], state["updates"], state["examples"]
        history, best = state["history"], state["best"]
        restore_rng(state["rng"])
    else:
        current = path.control_state()
        path.load_control_state(path.linear_control_state())
        initial_profile = path_profile(
            path, selection, cfg["device"], cfg["selection_points"]
        )
        initial_score = score(initial_profile, 0)
        best = artifact(path, cfg, pair, family, restart, 0.0, 0, initial_score)
        path.load_control_state(current)
        history.append(
            dict(pass_number=0, updates=0, examples=0, score=list(initial_score))
        )

    def recover():
        save(
            recovery_path,
            {
                "controls": path.control_state(),
                "optimizer": optimizer.state_dict(),
                "pass": completed,
                "updates": updates,
                "examples": examples,
                "history": history,
                "best": best,
                "rng": rng_state(),
            },
        )
        write_json(history_path, history)

    started = time.monotonic()
    while completed < cfg["fit_passes"]:
        lr = learning_rate(cfg["fit_lr"], completed, cfg["fit_passes"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        loader = data.loader(
            cfg["fit_subset"],
            augment=True,
            batch_size=cfg["fit_batch_size"],
            order_seed=seed_for(cfg, pair["id"], f"{family}-pass", completed),
        )
        path.train()
        for x, y in loader:
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            t = torch.rand((), device=cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits, path_l2 = path.forward_with_l2(x, t)
            nll = torch.nn.functional.cross_entropy(logits, y)
            loss = nll + 0.5 * cfg["path_weight_decay"] * path_l2
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite path-training objective.")
            loss.backward()
            optimizer.step()
            updates += 1
            examples += len(y)
        completed += 1
        if completed % cfg["validation_interval"] == 0 or completed == cfg["fit_passes"]:
            validation = path_profile(
                path, selection, cfg["device"], cfg["selection_points"]
            )
            candidate_score = score(validation, completed)
            if candidate_score < tuple(best["selected_score"]):
                best = artifact(
                    path, cfg, pair, family, restart, noise, completed, candidate_score
                )
            row = dict(
                pass_number=completed,
                updates=updates,
                examples=examples,
                lr=lr,
                score=list(candidate_score),
                best_score=best["selected_score"],
                elapsed_session_seconds=time.monotonic() - started,
            )
            history.append(row)
            print(row, flush=True)
            recover()
        if getattr(stop, "requested", False):
            recover()
            raise StopRequested("Interrupted after a complete path-training pass.")
    path.load_control_state(best["controls"])
    best["geometry"] = path_geometry(path)
    save(output_path, best)
    write_json(history_path, history)
    recovery_path.unlink(missing_ok=True)
    return dict(
        pair=pair["id"],
        family=family,
        restart=restart,
        selected_pass=best["selected_pass"],
        score=best["selected_score"],
        updates=updates,
        examples=examples,
    )
