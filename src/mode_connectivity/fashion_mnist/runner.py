"""Prepare Fashion-MNIST data or train one independent paper endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .model import make_model, permutation_spec
from .protocol import StopFlag, prepare, verify_protocol
from .training import train


def validate_config(cfg):
    if int(cfg["hidden_layers"]) != 10 or int(cfg["hidden_width"]) != 512:
        raise ValueError("The paper fixes 10 hidden layers of width 512")
    if int(cfg["epochs"]) <= 0 or int(cfg["train_batch_size"]) <= 0:
        raise ValueError("epochs and train_batch_size must be positive")
    checkpoints = [int(value) for value in cfg["checkpoints"]]
    if checkpoints != sorted(set(checkpoints)) or checkpoints[0] != 0:
        raise ValueError("checkpoints must be ordered, unique, and include epoch zero")
    if checkpoints[-1] != int(cfg["epochs"]):
        raise ValueError("the final epoch must be checkpointed")
    seeds = sum(cfg["seed_pairs"], [])
    if len(seeds) != len(set(seeds)) or any(len(pair) != 2 for pair in cfg["seed_pairs"]):
        raise ValueError("seed_pairs must contain six distinct endpoints in three pairs")
    if int(cfg["validation_size"]) >= 60_000:
        raise ValueError("validation_size must leave training examples")
    net = make_model(cfg)
    if set(net.state_dict()) != set(permutation_spec(cfg).axes_to_perm):
        raise ValueError("MLP parameters and permutation specification disagree")


def smoke(cfg):
    net = make_model(cfg)
    output = net(torch.zeros(2, 1, 28, 28))
    if tuple(output.shape) != (2, 10):
        raise RuntimeError("Unexpected FashionMLP output shape")
    return dict(
        parameters=sum(parameter.numel() for parameter in net.parameters()),
        output_shape=list(output.shape),
    )


def load_config(manifest, overrides):
    if manifest:
        if overrides:
            raise ValueError("A manifest cannot be combined with Hydra overrides")
        return json.loads(Path(manifest).read_text())["config"]
    from mode_connectivity.common.hydra_compat import compose_experiment_config
    from omegaconf import OmegaConf
    return OmegaConf.to_container(compose_experiment_config(
        default_config_name="fashion_mnist/default", caller_file=__file__, argv=overrides,
    ), resolve=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "smoke", "train"))
    parser.add_argument("--manifest")
    parser.add_argument("--seed", type=int)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args.manifest, overrides)
    validate_config(cfg)
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    if args.operation == "prepare":
        prepare(cfg)
        return
    if args.operation == "smoke":
        print(smoke(cfg))
        return
    verify_protocol(cfg)
    if args.seed not in sum(cfg["seed_pairs"], []):
        parser.error("train requires a configured --seed")
    print(train(cfg, args.seed, StopFlag()))


if __name__ == "__main__":
    main()
