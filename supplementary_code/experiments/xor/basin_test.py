"""Config-backed runner for the XOR same-function basin experiment.

If no CLI arguments are provided, the module loads a default YAML config and
forwards its argument list to the retained implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from experiments.xor._cli import invoke_main
from mode_connectivity.xor import xor_experiment


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs/experiments/xor/runners/basin_test.yaml"


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv and DEFAULT_CONFIG.exists():
        cfg = OmegaConf.load(DEFAULT_CONFIG)
        argv = list(cfg.get("argv", []))
    invoke_main(xor_experiment.main, argv)


if __name__ == "__main__":
    main()
