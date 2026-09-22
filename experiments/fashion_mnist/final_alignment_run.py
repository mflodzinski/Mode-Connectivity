"""Run final-endpoint Fashion-MNIST alignment tasks inside Slurm jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mode_connectivity.fashion_mnist.final_alignment import (
    evaluate_final_alignment,
    optimize_sinkhorn,
    optimize_sinkhorn_scales,
    report_final_alignment,
    run_weight_matching,
)
from mode_connectivity.fashion_mnist.protocol import StopFlag, verify_protocol
from mode_connectivity.fashion_mnist.runner import validate_config


def validate_final_alignment_config(cfg):
    validate_config(cfg)
    for key in [
        "sinkhorn_passes",
        "sinkhorn_iters",
        "sinkhorn_min_passes",
        "sinkhorn_patience",
    ]:
        if int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    for key in ["sinkhorn_lr", "sinkhorn_l", "sinkhorn_tau"]:
        if float(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if int(cfg["sinkhorn_min_passes"]) > int(cfg["sinkhorn_passes"]):
        raise ValueError("sinkhorn_min_passes cannot exceed sinkhorn_passes.")


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
    parser.add_argument("operation", choices=["base", "scale_evaluate", "report"])
    parser.add_argument("--manifest")
    parser.add_argument("--replicate", type=int)
    args, overrides = parser.parse_known_args()
    cfg = load_config(args, overrides)
    validate_final_alignment_config(cfg)
    torch.set_num_threads(int(cfg.get("torch_threads", 1)))
    verify_protocol(cfg)
    if args.operation == "report":
        print(report_final_alignment(cfg))
        return
    if args.replicate is None:
        parser.error(f"{args.operation} requires --replicate")
    if not 0 <= args.replicate < len(cfg["seed_pairs"]):
        parser.error("replicate is outside the configured seed-pair range")
    stop = StopFlag()
    if args.operation == "base":
        print(run_weight_matching(cfg, args.replicate, stop))
        result = optimize_sinkhorn(cfg, args.replicate, stop)
        print(
            dict(
                transform=result["transform"],
                completed=result["completed"],
                score=result["score"],
            )
        )
    else:
        result = optimize_sinkhorn_scales(cfg, args.replicate, stop)
        print(
            dict(
                transform=result["transform"],
                completed=result["completed"],
                score=result["score"],
            )
        )
        payload = evaluate_final_alignment(cfg, args.replicate, stop)
        print(dict(replicate=args.replicate, profiles=len(payload["profiles"])))


if __name__ == "__main__":
    main()
