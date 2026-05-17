"""Evaluate a trained Garipov curve with the vendored upstream evaluator.

The runner resolves the checkpoint and output paths from config and then calls
the retained ``dnn-mode-connectivity`` evaluation script with those values.
"""

import hydra
from omegaconf import DictConfig
from mode_connectivity.curves.runners import run_curve_evaluation

@hydra.main(
    version_base=None,
    config_path="../../configs/experiments",
    config_name="curves/runs/curve_seed0_seed1_reg",
)
def main(cfg: DictConfig):
    run_curve_evaluation(cfg=cfg, output_root_key="output_root", num_points=61)

if __name__ == "__main__":
    main()
