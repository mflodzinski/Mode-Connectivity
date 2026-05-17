#!/usr/bin/env python3
"""Launch polygon-chain fitting constrained to a random affine plane.

The runner reuses the shared curve orchestration layer and injects the
random-plane projection flags that define the codimension experiments.
"""

import hydra
from omegaconf import DictConfig

from mode_connectivity.common.utils import set_global_seed
from mode_connectivity.curves.runners import (
    build_curve_training_command,
    log_endpoint_distance,
    resolve_checkpoint_path,
    run_training_command,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/experiments",
    config_name="curves/runs/random_plane_midpoint_seed0_seed1",
)
def main(cfg: DictConfig):
    seed = cfg.get('seed', 0)
    set_global_seed(seed)
    anchor_type = "random anchor" if cfg.get('random_anchor', False) else "midpoint"
    run_dir = str(cfg.output_root)
    endpoint0 = resolve_checkpoint_path(str(cfg.endpoint0))
    endpoint1 = resolve_checkpoint_path(str(cfg.endpoint1))
    log_endpoint_distance(
        output_dir=run_dir,
        left_label=str(cfg.endpoint0),
        left_path=endpoint0,
        right_label=str(cfg.endpoint1),
        right_path=endpoint1,
        title=f"RANDOM PLANE OPTIMIZATION ({anchor_type}, seed={cfg.random_plane_seed})",
    )
    extra_flags = ["--project_random_plane"]
    if cfg.get("random_anchor", False):
        extra_flags.append("--random_anchor")
    cmd = build_curve_training_command(
        cfg=cfg,
        output_dir=run_dir,
        endpoint0=endpoint0,
        endpoint1=endpoint1,
        curve_type="PolyChain",
        num_bends=3,
        include_training_hparams=True,
        extra_flags=extra_flags,
        extra_kv={
            "random_plane_seed": int(cfg.random_plane_seed),
            "random_plane_codim": int(cfg.get("random_plane_codim", 1)),
        },
    )
    run_training_command(
        cmd,
        cfg=cfg,
        run_name=f"randomplane_{anchor_type}_{cfg.model}_{cfg.get('experiment_name', 'seed0-seed1')}",
    )


if __name__ == "__main__":
    main()
