"""CLI and orchestration for the Fashion-MNIST connectivity benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .alignment import find_permutation, refine_scales
from .evaluation import (
    evaluate_linear_pair,
    evaluate_nonlinear_pair,
    evaluate_pair,
)
from .model import make_model, permutation_spec
from .nonlinear import fit_nonlinear
from .protocol import StopFlag, checkpoint, prepare, verify_protocol
from .reporting import report
from .training import train


def validate_config(cfg):
    if int(cfg["hidden_layers"]) != 10 or int(cfg["hidden_width"]) != 512:
        raise ValueError(
            "The paper protocol fixes exactly 10 hidden layers of width 512."
        )
    if (
        cfg["stages"] != sorted(set(cfg["stages"]))
        or cfg["stages"][-1] != cfg["epochs"]
    ):
        raise ValueError("stages must be ordered, unique, and end at epochs.")
    if 0 not in cfg["checkpoints"] or not set(cfg["stages"]).issubset(
        cfg["checkpoints"]
    ):
        raise ValueError(
            "Initialization and every analysis stage must be checkpointed."
        )
    seeds = sum(cfg["seed_pairs"], [])
    if len(seeds) != len(set(seeds)) or any(
        len(pair) != 2 for pair in cfg["seed_pairs"]
    ):
        raise ValueError("Seed pairs must contain distinct independent seeds.")
    for key in [
        "epochs",
        "train_batch_size",
        "eval_batch_size",
        "alignment_batch_size",
        "scale_passes",
        "nonlinear_passes",
        "nonlinear_batch_size",
        "validation_interval",
        "eval_points",
    ]:
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if cfg["scale_passes"] % cfg["validation_interval"]:
        raise ValueError("scale_passes must be divisible by validation_interval.")
    if cfg["nonlinear_passes"] % cfg["validation_interval"]:
        raise ValueError("nonlinear_passes must be divisible by validation_interval.")
    if cfg["validation_size"] >= 60000 or cfg["test_eval_size"] > 10000:
        raise ValueError("Subset sizes exceed Fashion-MNIST.")
    if cfg["selection_size"] > cfg["validation_size"]:
        raise ValueError("selection_size must fit inside validation_size.")
    if cfg["alignment_size"] + cfg["train_eval_size"] > 60000 - cfg["validation_size"]:
        raise ValueError("alignment and train_eval subsets must fit and be disjoint.")
    net = make_model(cfg)
    if set(net.state_dict()) != set(permutation_spec(cfg).axes_to_perm):
        raise ValueError("MLP parameter names and permutation specification disagree.")


def smoke(cfg):
    net = make_model(cfg)
    output = net(torch.zeros(2, 1, 28, 28))
    if tuple(output.shape) != (2, 10):
        raise RuntimeError("Unexpected FashionMLP output shape.")
    if any(
        isinstance(module, (torch.nn.BatchNorm1d, torch.nn.Dropout))
        for module in net.modules()
    ):
        raise RuntimeError("The FashionMLP must remain normalization/dropout free.")
    return dict(
        parameters=sum(p.numel() for p in net.parameters()),
        output_shape=list(output.shape),
    )


def analyze(cfg, replicate: int, epoch: int, stop):
    for seed in cfg["seed_pairs"][replicate]:
        if not checkpoint(cfg, seed, epoch).exists():
            raise FileNotFoundError(checkpoint(cfg, seed, epoch))
    find_permutation(cfg, replicate, epoch, stop)
    refine_scales(cfg, replicate, epoch, stop)
    fit_nonlinear(cfg, replicate, epoch, stop)
    return evaluate_pair(cfg, replicate, epoch, stop)


def analyze_linear(cfg, replicate: int, epoch: int, stop):
    for seed in cfg["seed_pairs"][replicate]:
        if not checkpoint(cfg, seed, epoch).exists():
            raise FileNotFoundError(checkpoint(cfg, seed, epoch))
    find_permutation(cfg, replicate, epoch, stop)
    refine_scales(cfg, replicate, epoch, stop)
    return evaluate_linear_pair(cfg, replicate, epoch, stop)


def analyze_nonlinear(cfg, replicate: int, epoch: int, stop):
    for seed in cfg["seed_pairs"][replicate]:
        if not checkpoint(cfg, seed, epoch).exists():
            raise FileNotFoundError(checkpoint(cfg, seed, epoch))
    fit_nonlinear(cfg, replicate, epoch, stop)
    return evaluate_nonlinear_pair(cfg, replicate, epoch, stop)


def load_config(args, overrides):
    if args.manifest:
        if overrides:
            raise ValueError("A manifest cannot be combined with Hydra overrides.")
        return json.loads(Path(args.manifest).read_text())["config"]
    from mode_connectivity.common.hydra_compat import compose_experiment_config
    from omegaconf import OmegaConf

    return OmegaConf.to_container(
        compose_experiment_config(
            default_config_name="fashion_mnist/default",
            caller_file=__file__,
            argv=overrides,
        ),
        resolve=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation",
        choices=[
            "prepare",
            "smoke",
            "train",
            "analyze",
            "linear",
            "nonlinear",
            "report",
            "report_linear",
            "report_nonlinear",
            "all",
        ],
    )
    parser.add_argument("--manifest")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--replicate", type=int)
    parser.add_argument("--epoch", type=int)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args, overrides)
    validate_config(cfg)
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    if args.operation == "prepare":
        prepare(cfg)
        return
    if args.operation == "smoke":
        print(smoke(cfg))
        return
    if args.operation == "all":
        prepare(cfg)
        verify_protocol(cfg)
        stop = StopFlag()
        for seed in sum(cfg["seed_pairs"], []):
            train(cfg, seed, stop)
        for replicate in range(len(cfg["seed_pairs"])):
            for epoch in cfg["stages"]:
                analyze(cfg, replicate, epoch, stop)
        print(report(cfg))
        return
    verify_protocol(cfg)
    stop = StopFlag()
    if args.operation == "train":
        if args.seed is None:
            parser.error("train requires --seed")
        if args.seed not in sum(cfg["seed_pairs"], []):
            parser.error("seed is not part of the configured seed pairs")
        print(train(cfg, args.seed, stop))
    elif args.operation in ("analyze", "linear", "nonlinear"):
        if args.replicate is None or args.epoch is None:
            parser.error(f"{args.operation} requires --replicate and --epoch")
        operation = {
            "analyze": analyze,
            "linear": analyze_linear,
            "nonlinear": analyze_nonlinear,
        }[args.operation]
        print(operation(cfg, args.replicate, args.epoch, stop))
    elif args.operation in ("report_linear", "report_nonlinear"):
        from .stage_reporting import report_linear, report_nonlinear

        operation = {
            "report_linear": report_linear,
            "report_nonlinear": report_nonlinear,
        }[args.operation]
        print(operation(cfg))
    else:
        print(report(cfg))


if __name__ == "__main__":
    main()
