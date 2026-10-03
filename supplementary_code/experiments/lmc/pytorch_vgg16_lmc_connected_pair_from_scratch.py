"""Hydra entrypoint for the full shared-training benchmark from scratch.

The actual training logic lives in ``mode_connectivity.lmc``; this file only
binds that reusable workflow to the repo-level experiment surface.
"""

from __future__ import annotations

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.lmc.shared_training import run


def main() -> None:
    cfg = compose_experiment_config(
        default_config_name="lmc/runs/split_30",
        caller_file=__file__,
    )
    run(cfg)


if __name__ == "__main__":
    main()
