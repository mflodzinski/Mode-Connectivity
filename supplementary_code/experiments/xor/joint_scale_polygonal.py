"""Config-backed runner for joint scale and polygonal XOR paths."""

from __future__ import annotations

import sys
from pathlib import Path

from omegaconf import OmegaConf

from experiments.xor._cli import invoke_main
from mode_connectivity.xor import xor_joint_scale_polygonal


DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs/experiments/xor/runners/joint_scale_polygonal.yaml"
)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if not argv and DEFAULT_CONFIG.exists():
        cfg = OmegaConf.load(DEFAULT_CONFIG)
        argv = list(cfg.get("argv", []))
    invoke_main(xor_joint_scale_polygonal.main, argv)


if __name__ == "__main__":
    main()
