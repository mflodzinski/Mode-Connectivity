"""Endpoint training, with exact epoch-boundary recovery."""

import math
import time

import torch
import torch.nn.functional as F

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
    mixed_seed,
)


def git_rebasin_lr(step, steps_per_epoch, epochs, peak):
    """Optax warmup-cosine schedule used by the paper's CIFAR MLP runs."""
    warmup = steps_per_epoch
    total = epochs * steps_per_epoch
    if step < warmup:
        return 1e-6 + (peak - 1e-6) * step / warmup
    progress = min(max((step - warmup) / (total - warmup), 0.0), 1.0)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def git_rebasin_augment(images_u8, generator):
    """Torch equivalent of the pinned AugMax geometric CIFAR transform.

    It composes RandomSizedCrop(zoom_range=(.8, 1.2)), HorizontalFlip(), and
    Rotate() into one bilinear resampling with nearest-border extension, then
    applies ByteToFloat and AugMax's default ImageNet normalization.
    """
    if images_u8.ndim != 4 or tuple(images_u8.shape[1:]) != (3, 32, 32):
        raise ValueError("Expected an NCHW uint8 CIFAR-10 batch.")
    device = images_u8.device
    count = images_u8.shape[0]
    # Draw on CPU from a task-local generator so recovery and loader workers do
    # not influence the augmentation stream.
    zoom = torch.exp(
        torch.empty(count).uniform_(math.log(0.8), math.log(1.2), generator=generator)
    )
    limit = (32.0 * zoom - 32.0) / 2.0
    center = (2.0 * torch.rand(count, 2, generator=generator) - 1.0) * limit[:, None]
    flip = 1.0 - 2.0 * torch.bernoulli(
        torch.full((count,), 0.5), generator=generator
    )
    theta = torch.empty(count).uniform_(-math.pi / 6, math.pi / 6, generator=generator)
    cos, sin = theta.cos(), theta.sin()
    matrix = torch.zeros(count, 3, 3)
    matrix[:, 0, 0], matrix[:, 0, 1] = cos, sin
    matrix[:, 1, 0], matrix[:, 1, 1] = -sin, cos
    matrix[:, 2, 2] = 1.0
    horizontal = torch.eye(3).repeat(count, 1, 1)
    horizontal[:, 1, 1] = flip
    crop = torch.eye(3).repeat(count, 1, 1)
    crop[:, 0, 0] = 1.0 / zoom
    crop[:, 1, 1] = 1.0 / zoom
    crop[:, 0, 2] = center[:, 0] / zoom
    crop[:, 1, 2] = center[:, 1] / zoom
    matrix = (crop @ horizontal @ matrix).to(device)
    axis = torch.arange(32, device=device, dtype=torch.float32) - 15.5
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    coordinates = torch.stack([yy, xx, torch.ones_like(xx)]).reshape(1, 3, -1)
    source = matrix @ coordinates
    source_y = source[:, 0].reshape(count, 32, 32) + 15.5
    source_x = source[:, 1].reshape(count, 32, 32) + 15.5
    grid = torch.stack(
        [2.0 * source_x / 31.0 - 1.0, 2.0 * source_y / 31.0 - 1.0], dim=-1
    )
    images = F.grid_sample(
        images_u8.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ) / 255.0
    mean = images.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = images.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
    return (images - mean) / std


def train(cfg, seed, stop):
    directory = root(cfg) / "endpoints" / str(seed)
    directory.mkdir(parents=True, exist_ok=True)
    recovery = directory / "recovery.pt"
    seed_all(seed)
    git_recipe = cfg.get("training_recipe") == "git_rebasin_cifar10_mlp"
    data = Data(cfg, allow_test=git_recipe and cfg.get("test_monitor_each_epoch", True))
    net = model(cfg).to(cfg["device"])
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
    train_loader = None
    val_loader = None if git_recipe else data.loader("validation")
    test_loader = data.loader("test_full") if git_recipe else None
    steps_per_epoch = (
        len(data.subsets["indices"]["train"]) // cfg["train_batch_size"]
        if git_recipe
        else None
    )
    for epoch in range(first + 1, cfg["epochs"] + 1):
        stop.check()
        start = time.monotonic()
        if git_recipe:
            train_loader = data.loader(
                "train",
                raw=True,
                batch_size=cfg["train_batch_size"],
                order_seed=mixed_seed(seed, f"epoch-{epoch - 1}"),
                worker_seed=mixed_seed(seed, f"loader-{epoch - 1}"),
            )
        elif train_loader is None:
            train_loader = data.loader(
                "train", augment=True, shuffle=True, batch_size=cfg["train_batch_size"]
            )
        epoch_lr = None
        net.train()
        loss_sum, wrong, count = 0.0, 0, 0
        for batch_index, (x, y) in enumerate(train_loader):
            stop.check()  # Recovery is the previous complete epoch, including RNG.
            x, y = x.to(cfg["device"]), y.to(cfg["device"])
            if git_recipe:
                augmentation_seed = (
                    seed
                    if cfg.get("augmentation_seed_mode", "run") == "run"
                    else 0
                )
                augmentation_rng = torch.Generator().manual_seed(
                    mixed_seed(
                        augmentation_seed,
                        f"batch_rngs-{epoch - 1}-{batch_index}",
                    )
                )
                x = git_rebasin_augment(x, augmentation_rng)
                epoch_lr = git_rebasin_lr(
                    (epoch - 1) * steps_per_epoch + batch_index,
                    steps_per_epoch,
                    cfg["epochs"],
                    cfg["lr"],
                )
            else:
                epoch_lr = cfg["lr"] * 0.5 ** ((epoch - 1) // cfg["lr_step"])
            for group in optimizer.param_groups:
                group["lr"] = epoch_lr
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
        validation = None if git_recipe else evaluate(net, val_loader, cfg["device"])
        test = evaluate(net, test_loader, cfg["device"]) if git_recipe else None
        row = dict(
            epoch=epoch,
            official_checkpoint=(epoch - 1) if git_recipe else None,
            lr=epoch_lr,
            train_loss=loss_sum / count,
            train_error=100 * wrong / count,
            validation=validation,
            test_monitor=test,
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
