import os
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/curves",
    config_name="vgg16_curve_seed0-seed1_reg",
)
def main(cfg: DictConfig):
    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    eval_script = os.path.join(repo_root, "eval_curve.py")

    curve_checkpoint = to_absolute_path(
        os.path.join(cfg.output_root, f"checkpoint-{cfg.epochs}.pt")
    )

    # Save evaluation results to evaluations/ directory
    eval_dir = to_absolute_path(
        cfg.output_root.replace('/checkpoints', '/evaluations')
    )
    os.makedirs(eval_dir, exist_ok=True)

    cmd = [
        "python",
        eval_script,
        "--dir",
        eval_dir,
        "--dataset",
        cfg.dataset,
        "--data_path",
        cfg.data_path,
        "--transform",
        cfg.transform,
        "--model",
        cfg.model,
        "--curve",
        cfg.curve,
        "--num_bends",
        str(cfg.num_bends),
        "--ckpt",
        curve_checkpoint,
        "--num_points",
        "61",  # Evaluate at 61 points along the curve
    ]

    if cfg.use_test:
        cmd.append("--use_test")

    print("Evaluating curve:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
