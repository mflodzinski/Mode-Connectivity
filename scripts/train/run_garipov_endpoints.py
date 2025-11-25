import os
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.utils import set_global_seed

@hydra.main(
    version_base=None,
    config_path="../../configs/garipov",
    config_name="vgg16_endpoints",
)
def main(cfg: DictConfig):
    set_global_seed(0)  # just for wrapper

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    for seed in cfg.seeds:
        run_dir = to_absolute_path(
            os.path.join(cfg.output_root, f"{cfg.model}_seed{seed}")
        )
        os.makedirs(run_dir, exist_ok=True)

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

        # Add seed
        cmd += ["--seed", str(seed)]

        # Add wandb flags
        if cfg.use_wandb:
            run_name = f"garipov_{cfg.model}_endpoint_seed{seed}"
            cmd.append("--wandb")
            cmd += ["--wandb_project", cfg.project_name]
            cmd += ["--wandb_name", run_name]

        print("Running:", " ".join(cmd))

        # Training script now handles wandb internally, so no need to wrap here
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()