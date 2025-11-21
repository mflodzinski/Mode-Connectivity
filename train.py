import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import wandb

from src.utils import set_global_seed, get_git_commit
from src import train_loop, data, models  # you'd implement these

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    set_global_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    cfg.run_name = f"{cfg.experiment.name}-{cfg.model.name}-seed={cfg.seed}"

    wandb.init(
        project=cfg.project_name,
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    # log repo commits
    wandb.config["thesis_commit"] = get_git_commit(".")
    wandb.config["torch_version"] = torch.__version__

    train_loader, val_loader = data.build_dataloaders(cfg)
    model = models.build_model(cfg.model).to(device)

    best_val_acc = train_loop.run_training(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
    )

    wandb.summary["best_val_acc"] = best_val_acc
    wandb.finish()

if __name__ == "__main__":
    main()


# poetry run train experiment=baseline model=mlp seed=0
# poetry run train -m seed=0,1,2,3