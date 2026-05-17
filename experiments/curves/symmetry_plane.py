"""Launch polygon-chain fitting constrained to the endpoint symmetry plane.

This runner reuses the shared curve orchestration layer and enables the
projection flag that keeps the interior point on the symmetry plane.
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
    config_name="curves/runs/symmetry_plane_seed0_seed1",
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
        title="SYMMETRY PLANE OPTIMIZATION",
    )
    cmd = build_curve_training_command(
        cfg=cfg,
        output_dir=run_dir,
        endpoint0=endpoint0,
        endpoint1=endpoint1,
        curve_type="PolyChain",
        num_bends=3,
        include_training_hparams=True,
        extra_flags=["--project_symmetry_plane"],
    )
    run_training_command(cmd, cfg=cfg, run_name=f"symplane_{cfg.model}_{cfg.experiment_name}")


if __name__ == "__main__":
    main()
