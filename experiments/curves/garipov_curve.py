"""Launch standard Garipov curve training between two fixed endpoints.

This runner logs endpoint distance diagnostics, builds the upstream training
command, and executes it with the resolved Hydra configuration.
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
    config_name="curves/runs/curve_seed0_seed1_reg",
)
def main(cfg: DictConfig):
    seed = cfg.get('seed', 0)
    set_global_seed(seed)
    run_dir = str(cfg.output_root)
    endpoint0 = resolve_checkpoint_path(str(cfg.endpoint0))
    endpoint1 = resolve_checkpoint_path(str(cfg.endpoint1))
    log_endpoint_distance(
        output_dir=run_dir,
        left_label=str(cfg.endpoint0),
        left_path=endpoint0,
        right_label=str(cfg.endpoint1),
        right_path=endpoint1,
        title="L2 DISTANCE BETWEEN ENDPOINTS",
    )
    cmd = build_curve_training_command(
        cfg=cfg,
        output_dir=run_dir,
        endpoint0=endpoint0,
        endpoint1=endpoint1,
    )
    run_training_command(cmd, cfg=cfg, run_name=f"garipov_{cfg.model}_curve_{cfg.curve}")

if __name__ == "__main__":
    main()
