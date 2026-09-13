"""Endpoint training, with exact epoch-boundary recovery and true validation."""

import time

import torch

from .geometry import model, evaluate
from .protocol import (
    Data,
    checkpoint,
    root,
    load,
    save,
    seed_all,
    rng_state,
    restore_rng,
    write_json,
)


def train(cfg, seed, stop):
    directory = root(cfg) / "endpoints" / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    recovery = directory / "recovery.pt"
    seed_all(seed)
    data = Data(cfg)
    net = model().to(cfg["device"])
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
    )
    first, history = 0, []
    if recovery.exists():
        state = load(recovery)
        net.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        first, history = state["epoch"], state["history"]
        # An interrupted save may have persisted recovery before the scientific copy.
        if first in cfg["checkpoints"] and not checkpoint(cfg, seed, first).exists():
            save(
                checkpoint(cfg, seed, first),
                dict(state_dict=state["state_dict"], epoch=first, seed=seed),
            )
        restore_rng(state["rng"])
        del state
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
    train_loader = data.loader(
        "train", augment=True, shuffle=True, batch_size=cfg["train_batch_size"]
    )
    val_loader = data.loader("validation")
    for epoch in range(first + 1, cfg["epochs"] + 1):
        stop.check()
        start = time.monotonic()
        lr = cfg["lr"] * 0.5 ** ((epoch - 1) // cfg["lr_step"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        net.train()
        loss_sum, wrong, count = 0.0, 0, 0
        for x, y in train_loader:
            stop.check()  # Recovery is the previous complete epoch, including RNG.
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            optimizer.zero_grad(set_to_none=True)
            logits = net(x)
            loss = torch.nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite endpoint training loss.")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            wrong += (logits.argmax(1) != y).sum().item()
            count += len(y)
        validation = evaluate(net, val_loader, cfg["device"])
        row = dict(
            epoch=epoch,
            lr=lr,
            train_loss=loss_sum / count,
            train_error=100 * wrong / count,
            validation=validation,
            seconds=time.monotonic() - start,
        )
        history.append(row)
        save(
            recovery,
            dict(
                state_dict=net.state_dict(),
                optimizer=optimizer.state_dict(),
                epoch=epoch,
                seed=seed,
                history=history,
                rng=rng_state(),
            ),
        )
        if epoch in cfg["checkpoints"]:
            save(
                checkpoint(cfg, seed, epoch),
                dict(state_dict=net.state_dict(), epoch=epoch, seed=seed),
            )
        write_json(directory / "history.json", history)
        print(row, flush=True)
        stop.check()
    return dict(
        seed=seed,
        epochs=cfg["epochs"],
        training_seconds=sum(r["seconds"] for r in history),
    )
