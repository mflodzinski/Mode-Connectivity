"""Resumable endpoint training for the Fashion-MNIST MLP."""

from __future__ import annotations

import time

import torch

from .model import make_model
from .profiles import evaluate
from .protocol import (
    Data,
    checkpoint,
    load,
    mixed_seed,
    restore_rng,
    rng_state,
    root,
    save,
    seed_all,
    write_json,
)


def train(cfg, seed: int, stop):
    directory = root(cfg) / "endpoints" / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    recovery = directory / "recovery.pt"
    seed_all(seed)
    data = Data(cfg)
    net = make_model(cfg).to(cfg["device"])
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=float(cfg["lr"]),
        betas=tuple(float(v) for v in cfg["adam_betas"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    first, history = 0, []
    if recovery.exists():
        state = load(recovery)
        net.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        first, history = int(state["epoch"]), state["history"]
        restore_rng(state["rng"])
    else:
        state = dict(
            state_dict=net.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=0,
            seed=seed,
            history=[],
            rng=rng_state(),
        )
        save(recovery, state)
        save(
            checkpoint(cfg, seed, 0),
            dict(state_dict=net.state_dict(), epoch=0, seed=seed),
        )
    validation = data.loader("validation")
    for epoch in range(first + 1, int(cfg["epochs"]) + 1):
        stop.check()
        started = time.monotonic()
        loader = data.loader(
            "train",
            batch_size=int(cfg["train_batch_size"]),
            order_seed=mixed_seed(seed, f"fashion-epoch-{epoch}"),
        )
        net.train()
        total, wrong, count = 0.0, 0, 0
        for x, y in loader:
            stop.check()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite endpoint training loss.")
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            wrong += (logits.argmax(1) != y).sum().item()
            count += len(y)
        row = dict(
            epoch=epoch,
            train_loss=total / count,
            train_error=100.0 * wrong / count,
            validation=evaluate(net, validation, cfg["device"]),
            seconds=time.monotonic() - started,
        )
        history.append(row)
        state = dict(
            state_dict=net.state_dict(),
            optimizer=optimizer.state_dict(),
            epoch=epoch,
            seed=seed,
            history=history,
            rng=rng_state(),
        )
        save(recovery, state)
        if epoch in cfg["checkpoints"]:
            save(
                checkpoint(cfg, seed, epoch),
                dict(state_dict=net.state_dict(), epoch=epoch, seed=seed),
            )
        write_json(directory / "history.json", history)
        print(row, flush=True)
    return dict(seed=seed, epochs=cfg["epochs"], final=history[-1])
