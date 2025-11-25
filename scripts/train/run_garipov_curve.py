import os
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.utils import set_global_seed

@hydra.main(
    version_base=None,
    config_path="../../configs/garipov",
    config_name="vgg16_curve",
)
def main(cfg: DictConfig):
    set_global_seed(0)

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    run_dir = to_absolute_path(cfg.output_root)
    os.makedirs(run_dir, exist_ok=True)

    # Get absolute paths to endpoints
    endpoint0 = to_absolute_path(cfg.endpoint0)
    endpoint1 = to_absolute_path(cfg.endpoint1)

    cmd = [
        "python",
        train_script,
        "--dir",
        run_dir,
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
        "--init_start",
        endpoint0,
        "--fix_start",  # Fix the starting endpoint
        "--init_end",
        endpoint1,
        "--fix_end",  # Fix the ending endpoint
        "--epochs",
        str(cfg.epochs),
        "--lr",
        str(cfg.lr),
        "--wd",
        str(cfg.wd),
    ]

    # Only save intermediate checkpoints if requested
    if cfg.get("save_intermediate", True):
        cmd += ["--save_freq", str(cfg.save_freq)]

    if cfg.use_test:
        cmd.append("--use_test")

    if cfg.use_wandb:
        run_name = f"garipov_{cfg.model}_curve_{cfg.curve}"
        cmd.append("--wandb")
        cmd += ["--wandb_project", cfg.project_name]
        cmd += ["--wandb_name", run_name]

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
