"""Validation-selected Sinkhorn optimization, independent of test data."""

import time

import torch

from mode_connectivity.alignment.permutation_spec import vgg_features_permutation_spec
from mode_connectivity.alignment.weight_matching import weight_matching
from .geometry import (
    alignment_state,
    set_alignment,
    hard_artifact,
    make_rebasin,
    freeze,
    read_model,
    interpolated_logits,
    profile,
    equivalence,
)
from .protocol import (
    Data,
    checkpoint,
    pair_dir,
    load,
    save,
    seed_all,
    rng_state,
    restore_rng,
    write_json,
    artifact_provenance,
)


def weight_artifact(a, b, cfg):
    start = time.monotonic()
    with torch.no_grad():
        perm = weight_matching(
            vgg_features_permutation_spec(cfg["model"]),
            a.state_dict(),
            b.state_dict(),
            max_iter=cfg["wm_sweeps"],
            seed=cfg["alignment_seed"],
            silent=True,
        )
    return dict(weight_permutation=perm, seconds=time.monotonic() - start)


def better(score, best):
    return best is None or tuple(score) < tuple(best)


def optimize(cfg, replicate, ea, eb, phase, stop):
    directory = pair_dir(cfg, replicate, ea, eb)
    directory.mkdir(parents=True, exist_ok=True)
    seeds = cfg["seed_pairs"][replicate]
    seed_all(cfg["alignment_seed"])
    a = freeze(read_model(checkpoint(cfg, seeds[0], ea), cfg))
    b = freeze(read_model(checkpoint(cfg, seeds[1], eb), cfg))
    endpoint_paths = [checkpoint(cfg, seeds[0], ea), checkpoint(cfg, seeds[1], eb)]
    provenance = artifact_provenance(cfg, endpoint_paths)
    data = Data(cfg)
    val = data.loader("selection")
    opt_loader = data.loader(
        "alignment", shuffle=True, batch_size=cfg["alignment_batch_size"]
    )
    if phase == "base" and not (directory / "wm.pt").exists():
        wm = weight_artifact(a, b, cfg)
        wm.update(transform="weight_matching", **provenance)
        save(directory / "wm.pt", wm)
        stop.check()
    starting = None if phase == "base" else load(directory / "base.pt")
    scale = phase == "scale"
    pi = make_rebasin(b, cfg, scale=scale, artifact=starting)
    if scale:
        with torch.no_grad():
            for u in pi.u:
                if u is not None:
                    u.zero_()
    optimizer = torch.optim.AdamW(
        [p for p in pi.parameters() if p.requires_grad],
        lr=cfg["scale_lr"] if scale else cfg["base_lr"],
        weight_decay=cfg["alignment_weight_decay"],
    )
    recovery = directory / f"{phase}_recovery.pt"
    maximum = cfg["base_passes"] if phase == "base" else cfg["branch_passes"]
    completed, updates, examples, stale, reference = 0, 0, 0, 0, None
    best_score, best_state, history = None, None, []
    if recovery.exists():
        state = load(recovery)
        set_alignment(pi, state["alignment"])
        optimizer.load_state_dict(state["optimizer"])
        completed, updates, examples = (
            state["completed"],
            state["updates"],
            state["examples"],
        )
        best_score, best_state, history = (
            state["best_score"],
            state["best_state"],
            state["history"],
        )
        stale, reference = state["stale"], state["reference"]
        restore_rng(state["rng"])
        del state
    start = time.monotonic()

    def select(pass_number):
        nonlocal best_score, best_state, stale, reference
        pi.eval()
        with torch.no_grad():
            hard = pi()
            hard.eval()
            discrepancy = equivalence(b, hard, val, cfg["device"], cfg)
            curve = profile(
                a, hard, val, cfg["device"], cfg["validation_alphas"], stop=stop
            )
        score = [curve["loss"]["chord"], curve["loss"]["mean"], pass_number]
        if better(score, best_score):
            best_score, best_state = score, hard_artifact(pi)
        if reference is None or reference - score[0] > cfg["min_delta"]:
            stale, reference = 0, score[0]
        else:
            stale += 1
        history.append(
            dict(
                pass_number=pass_number,
                updates=updates,
                examples=examples,
                score=score,
                max_logit_difference=discrepancy,
                elapsed_session_seconds=time.monotonic() - start,
            )
        )
        save(
            recovery,
            dict(
                alignment=alignment_state(pi),
                optimizer=optimizer.state_dict(),
                completed=pass_number,
                updates=updates,
                examples=examples,
                best_score=best_score,
                best_state=best_state,
                history=history,
                stale=stale,
                reference=reference,
                rng=rng_state(),
            ),
        )
        write_json(directory / f"{phase}_history.json", history)
        print(history[-1], flush=True)

    if best_state is None:
        select(0)
    for pass_number in range(completed + 1, maximum + 1):
        if (
            phase == "base"
            and completed >= cfg["min_passes"]
            and stale >= cfg["patience"]
        ):
            break
        stop.check()
        pi.train()
        # Soft/hard selection mode and dropout mode are independent here.
        pi.reparamnet.model.eval()
        pi.reparamnet.output.eval()
        for x, y in opt_loader:
            stop.check()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            rebased = pi()
            rebased.eval()
            alpha = torch.rand((), device=cfg["device"])
            logits = interpolated_logits(a, rebased, alpha, x)
            loss = torch.nn.functional.cross_entropy(logits, y) + pi.scale_regularizer()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite alignment objective.")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in pi.parameters()
            ):
                raise FloatingPointError("Nonfinite alignment gradient.")
            optimizer.step()
            updates += 1
            examples += len(y)
        completed = pass_number
        if pass_number % cfg["validation_interval"] == 0 or pass_number == maximum:
            select(pass_number)
            stop.check()
    # The best artifact contains raw scores (for continuation) and hard permutations.
    best_state.update(
        scale=scale,
        score=best_score,
        completed=completed,
        updates=updates,
        examples=examples,
        history=history,
        transform=phase,
        **provenance,
    )
    save(directory / f"{phase}.pt", best_state)
    return dict(
        passes=completed, updates=updates, examples=examples, selected_score=best_score
    )
