"""Resume split training from a previously materialized shared checkpoint.

This runner handles the second stage of the shared-training benchmark by
loading the common checkpoint and finishing each branch with its split seed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from mode_connectivity.common.hydra_compat import compose_experiment_config
from mode_connectivity.lmc.shared_training import (
    build_loaders,
    build_model,
    cudnn,
    load_checkpoint,
    nn,
    seed_all,
    torch,
    train_range,
)


def run(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = False
    cudnn.deterministic = True

    output_root = Path(to_absolute_path(cfg.output_root))
    data_root = to_absolute_path(cfg.data_root)
    shared_checkpoint = Path(to_absolute_path(cfg.shared_checkpoint))

    print("=" * 72)
    print("PYTORCH-VGG16 SHARED-CHECKPOINT SPLIT TRAINING")
    print("=" * 72)
    print(f"Shared checkpoint: {shared_checkpoint}")
    print(f"Output root: {output_root}")
    print(f"Final epochs: {cfg.final_epochs}")
    print(f"Split seeds: {list(cfg.split_seeds)}")

    if not shared_checkpoint.exists():
        raise FileNotFoundError(f"Missing shared checkpoint: {shared_checkpoint}")

    if output_root.exists():
        print(f"Removing existing output directory: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for branch_idx, seed in enumerate(cfg.split_seeds):
        branch_dir = output_root / f"seed{branch_idx}"
        seed = int(seed)
        print("\n" + "-" * 72)
        print(f"Training branch {branch_idx} with seed {seed}")
        print("-" * 72)

        seed_all(seed)
        model = build_model(cfg.arch, device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=float(cfg.lr),
            momentum=float(cfg.momentum),
            weight_decay=float(cfg.wd),
        )
        criterion = nn.CrossEntropyLoss().to(device)
        start_epoch, best_prec1 = load_checkpoint(shared_checkpoint, model, optimizer)
        train_loader, test_loader = build_loaders(data_root, int(cfg.batch_size), int(cfg.workers))
        best_prec1 = train_range(
            run_dir=branch_dir,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            start_epoch=start_epoch,
            final_epochs=int(cfg.final_epochs),
            base_lr=float(cfg.lr),
            save_every=int(cfg.save_every),
            best_prec1=best_prec1,
            epoch_print_freq=int(cfg.epoch_print_freq),
        )
        print(f"Branch {branch_idx} final checkpoint: {branch_dir / f'checkpoint-{cfg.final_epochs}.pt'}")
        print(f"Branch {branch_idx} best val acc: {best_prec1:.3f}")


def main() -> None:
    cfg = compose_experiment_config(
        default_config_name="lmc/runs/resume_shared_checkpoint",
        caller_file=__file__,
    )
    run(cfg)


if __name__ == "__main__":
    main()
