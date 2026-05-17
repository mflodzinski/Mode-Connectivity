"""Hydra entrypoint for the full shared-training benchmark from scratch.

The actual training logic lives in ``mode_connectivity.lmc``; this file only
binds that reusable workflow to the repo-level experiment surface.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from mode_connectivity.lmc.shared_training import run


@hydra.main(
    version_base=None,
    config_path="../../configs/experiments",
    config_name="lmc/runs/split_30",
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
