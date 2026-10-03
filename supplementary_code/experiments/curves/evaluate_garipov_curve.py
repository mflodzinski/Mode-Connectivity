"""Evaluate a trained Garipov curve with the vendored upstream evaluator.

The runner resolves the checkpoint and output paths from config and then calls
the retained ``dnn-mode-connectivity`` evaluation script with those values.
"""

from omegaconf import DictConfig
from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.curves.runners import run_curve_evaluation

def run(cfg: DictConfig) -> None:
    run_curve_evaluation(cfg=cfg, output_root_key="output_root", num_points=61)


def main() -> None:
    cfg = compose_experiment_config(
        default_config_name="curves/runs/curve_seed0_seed1_reg",
        caller_file=__file__,
    )
    run(cfg)

if __name__ == "__main__":
    main()
