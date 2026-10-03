"""Resumable CIFAR-10 VGG endpoint training."""

from __future__ import annotations

import time

import torch

from .endpoint_utils import evaluate, model
from .protocol import (
    Data, checkpoint, load, restore_rng, rng_state, root, save, seed_all,
    write_json,
)


def train(cfg, seed, stop):
    directory = root(cfg) / "endpoints" / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    recovery = directory / "recovery.pt"
    seed_all(seed)
    data = Data(cfg)
    net = model(cfg).to(cfg["device"])
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=float(cfg["lr"]),
        momentum=float(cfg["momentum"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    first, history = 0, []
    if recovery.exists():
        state = load(recovery)
        net.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        first, history = int(state["epoch"]), state["history"]
        if first in cfg["checkpoints"] and not checkpoint(cfg, seed, first).exists():
            save(
                checkpoint(cfg, seed, first),
                dict(state_dict=state["state_dict"], epoch=first, seed=seed),
            )
        restore_rng(state["rng"])
    else:
        state = dict(
            state_dict=net.state_dict(), optimizer=optimizer.state_dict(),
            epoch=0, seed=seed, history=[], rng=rng_state(),
        )
        save(recovery, state)
        save(checkpoint(cfg, seed, 0), dict(state_dict=net.state_dict(), epoch=0, seed=seed))

    train_loader = data.loader(
        "train", augment=True, shuffle=True, batch_size=int(cfg["train_batch_size"])
    )
    validation_loader = data.loader("validation")
    for epoch in range(first + 1, int(cfg["epochs"]) + 1):
        stop.check()
        started = time.monotonic()
        learning_rate = float(cfg["lr"]) * 0.5 ** ((epoch - 1) // int(cfg["lr_step"]))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        net.train()
        loss_sum, wrong, count = 0.0, 0, 0
        for x, y in train_loader:
            stop.check()
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite endpoint training loss")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            wrong += (logits.argmax(1) != y).sum().item()
            count += len(y)
        row = dict(
            epoch=epoch,
            lr=learning_rate,
            train_loss=loss_sum / count,
            train_error=100.0 * wrong / count,
            validation=evaluate(net, validation_loader, cfg["device"]),
            seconds=time.monotonic() - started,
        )
        history.append(row)
        save(recovery, dict(
            state_dict=net.state_dict(), optimizer=optimizer.state_dict(),
            epoch=epoch, seed=seed, history=history, rng=rng_state(),
        ))
        if epoch in cfg["checkpoints"]:
            save(
                checkpoint(cfg, seed, epoch),
                dict(state_dict=net.state_dict(), epoch=epoch, seed=seed),
            )
        write_json(directory / "history.json", history)
        print(row, flush=True)
    return dict(
        seed=seed,
        epochs=int(cfg["epochs"]),
        training_seconds=sum(float(row["seconds"]) for row in history),
    )
