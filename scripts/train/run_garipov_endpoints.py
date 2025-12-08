import os
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.utils import set_global_seed

@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/endpoints",
    config_name="vgg16_endpoints",
)
def main(cfg: DictConfig):
    set_global_seed(0)  # just for wrapper

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")

    for seed in cfg.seeds:
        run_dir = to_absolute_path(
            os.path.join(cfg.output_root, f"seed{seed}")
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

        # Add early stopping parameters if enabled
        if cfg.get("early_stopping", False):
            cmd.append("--early_stopping")
            cmd += ["--patience", str(cfg.get("patience", 20))]
            cmd += ["--min_delta", str(cfg.get("min_delta", 0.0))]
            cmd.append("--split_test_from_train")

        # Add wandb flags
        if cfg.use_wandb:
            run_name = f"garipov_{cfg.model}_endpoint_seed{seed}"
            if cfg.get("early_stopping", False):
                run_name += "_early_stop"
            cmd.append("--wandb")
            cmd += ["--wandb_project", cfg.project_name]
            cmd += ["--wandb_name", run_name]

        print("Running:", " ".join(cmd))

        # Training script now handles wandb internally, so no need to wrap here
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()