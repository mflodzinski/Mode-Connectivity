"""Launch polygon-chain fitting between two endpoints.

This runner specializes the shared curve command builder to ``PolyChain`` and
adds the bookkeeping used by the polygon experiments in the thesis.
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
    config_name="curves/runs/polygon_seed0_mirror",
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
        title="POLYGON CHAIN OPTIMIZATION",
    )
    cmd = build_curve_training_command(
        cfg=cfg,
        output_dir=run_dir,
        endpoint0=endpoint0,
        endpoint1=endpoint1,
        curve_type="PolyChain",
        num_bends=3,
        include_training_hparams=True,
    )
    run_training_command(cmd, cfg=cfg, run_name=f"polygon_{cfg.model}_{cfg.experiment_name}")


if __name__ == "__main__":
    main()
