import os
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

@hydra.main(
    version_base=None,
    config_path="configs/garipov",
    config_name="vgg16_curve",
)
def main(cfg: DictConfig):
    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    eval_script = os.path.join(repo_root, "eval_curve.py")

    curve_checkpoint = to_absolute_path(
        os.path.join(cfg.output_root, f"checkpoint-{cfg.epochs}.pt")
    )

    cmd = [
        "python",
        eval_script,
        "--dir",
        to_absolute_path(cfg.output_root),
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
