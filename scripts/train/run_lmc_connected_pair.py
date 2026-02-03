"""Train LMC-connected pairs using shared initialization with split training.

This script creates pairs of models that are:
1. Both genuine SGD solutions (trained to convergence)
2. LMC-connected (low barrier, typically <1%)
3. Different points in weight space

Approach:
1. Use existing checkpoint as shared initialization (or train from scratch)
2. From w_shared, continue training with TWO different batch orderings:
   - Batch seed A → train to convergence → w_0
   - Batch seed B → train to convergence → w_1
"""

import os
import sys
import subprocess
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)

from src.utils import set_global_seed
from lib.core.training_commands import (
    add_wandb_args, add_seed_arg,
    add_save_freq_arg, add_optional_arg, print_and_format_command
)


def build_base_command_no_epochs(train_script: str, run_dir: str, cfg) -> list:
    """Build base training command without epochs (we'll add them separately)."""
    cmd = [
        "python",
        train_script,
        "--dir", run_dir,
        "--dataset", cfg.dataset,
        "--data_path", cfg.data_path,
        "--transform", cfg.transform,
        "--model", cfg.model,
        "--lr", str(cfg.lr),
        "--wd", str(cfg.wd),
    ]
    return cmd


@hydra.main(
    version_base=None,
    config_path="../../configs/garipov/endpoints",
    config_name="vgg16_lmc_connected_pair",
)
def main(cfg: DictConfig):
    set_global_seed(cfg.shared_seed)

    repo_root = to_absolute_path("external/dnn-mode-connectivity")
    train_script = os.path.join(repo_root, "train.py")
    output_root = to_absolute_path(cfg.output_root)

    # =========================================================================
    # Get shared checkpoint (use existing or train from scratch)
    # =========================================================================
    # Use existing checkpoint instead of training from scratch
    shared_checkpoint = to_absolute_path(cfg.shared_checkpoint)

    print("=" * 70)
    print("Using existing shared checkpoint")
    print(f"  Checkpoint: {shared_checkpoint}")
    print(f"  Starting epoch: {cfg.shared_epochs}")
    print(f"  Final epochs: {cfg.final_epochs}")
    print("=" * 70)

    if not os.path.exists(shared_checkpoint):
        raise FileNotFoundError(
            f"Shared checkpoint not found: {shared_checkpoint}\n"
            f"Please verify the checkpoint path exists."
        )

    # =========================================================================
    # Stage 2: Split training from shared checkpoint
    # =========================================================================
    print("\n" + "=" * 70)
    print("STAGE 2: Split training from shared checkpoint")
    print(f"  Resume from: {shared_checkpoint}")
    print(f"  Final epochs: {cfg.final_epochs}")
    print(f"  Split seeds: {cfg.split_seeds}")
    print("=" * 70)

    for i, seed in enumerate(cfg.split_seeds):
        print(f"\n--- Training w_{i} with seed {seed} ---")

        run_dir = os.path.join(output_root, f"seed{i}")
        os.makedirs(run_dir, exist_ok=True)

        # Build training command
        # Resume handles LR schedule correctly (starts from checkpoint epoch + 1)
        cmd = build_base_command_no_epochs(train_script, run_dir, cfg)
        cmd += ["--epochs", str(cfg.final_epochs)]
        add_seed_arg(cmd, seed)  # Different seed for different batch ordering
        cmd += ["--resume", shared_checkpoint]  # Resume from shared checkpoint
        add_save_freq_arg(cmd, cfg)
        add_optional_arg(cmd, cfg, 'use_test', '--use_test', is_flag=True)

        # Add WandB logging
        run_name = f"garipov_{cfg.model}_lmc_split_seed{seed}"
        add_wandb_args(cmd, cfg, run_name)

        print_and_format_command(cmd)
        subprocess.run(cmd, check=True)

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Shared checkpoint: {shared_checkpoint}")
    for i, seed in enumerate(cfg.split_seeds):
        final_ckpt = os.path.join(
            output_root, f"seed{i}",
            f"checkpoint-{cfg.final_epochs}.pt"
        )
        print(f"w_{i} checkpoint: {final_ckpt}")
    print("\nThese endpoints should be LMC-connected (low barrier).")
    print("Use curve training to verify connectivity.")


if __name__ == "__main__":
    main()
