import os
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import wandb

from src.utils import set_global_seed, get_git_commit

@hydra.main(
    version_base=None,
    config_path="configs/garipov",
    config_name="vgg16_endpoints",
)
def main(cfg: DictConfig):
    set_global_seed(0)  # just for wrapper

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")
    garipov_commit = get_git_commit(repo_root)

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
        if cfg.use_test:
            cmd.append("--use_test")

        # if your fork has a --seed / --wandb flag, add:
        # cmd += ["--seed", str(seed)]
        # if cfg.use_wandb:
        #     cmd.append("--wandb")

        print("Running:", " ".join(cmd))

        # Optional: wrap in a wandb run just to track metadata & final results
        if cfg.use_wandb:
            run_name = f"garipov_vgg16_endpoint_seed{seed}"
            wandb.init(
                project=cfg.project_name,
                name=run_name,
                config={
                    "external_repo": "dnn-mode-connectivity",
                    "external_commit": garipov_commit,
                    "seed": seed,
                    "dataset": cfg.dataset,
                    "model": cfg.model,
                },
            )

        subprocess.run(cmd, check=True)

        if cfg.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main()


# poetry run run_garipov_endpoints
# # or
# poetry run run_garipov_endpoints seeds=[0,1,2]