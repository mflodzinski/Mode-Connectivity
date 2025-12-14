"""
Script to train multiple Bezier curves with different random seeds.
This allows testing the effect of stochasticity in batch shuffling,
SGD dynamics, and data augmentation on the resulting curves.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)

from lib.core.training_commands import print_and_format_command


def run_curve_training(config_name, seed, base_output_dir=None):
    """
    Run a single curve training with the specified seed.

    Args:
        config_name: Name of the base config file (without .yaml)
        seed: Random seed to use
        base_output_dir: Optional base directory for outputs (will append _seedX)
    """
    # Construct the command with Hydra overrides
    cmd = [
        "python",
        "scripts/train/run_garipov_curve.py",
        f"--config-name={config_name}",
        f"seed={seed}",
    ]

    # Override output directory if base_output_dir is provided
    if base_output_dir:
        output_path = f"{base_output_dir}_seed{seed}/checkpoints"
        cmd.append(f"output_root={output_path}")
        cmd.append(f"experiment_name={Path(config_name).stem}_seed{seed}")

    print("\n" + "="*80)
    print(f"STARTING TRAINING RUN WITH SEED={seed}")
    print("="*80)
    print_and_format_command(cmd)
    print("="*80 + "\n")

    # Run the command
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        print(f"\n⚠️  WARNING: Training with seed={seed} failed with return code {result.returncode}")
        return False
    else:
        print(f"\n✅ Training with seed={seed} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Train multiple Bezier curves with different random seeds"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        required=True,
        help="Base config name (e.g., vgg16_curve_seed0-seed1_noreg)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 42, 123],
        help="List of seeds to use (default: 0 42 123)"
    )
    parser.add_argument(
        "--base-output-dir",
        type=str,
        default=None,
        help="Base output directory (will append _seedX). If not specified, uses config default."
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("MULTI-SEED BEZIER CURVE TRAINING")
    print("="*80)
    print(f"Config: {args.config_name}")
    print(f"Seeds: {args.seeds}")
    print(f"Base output dir: {args.base_output_dir or 'From config'}")
    print("="*80 + "\n")

    # Run training for each seed
    results = {}
    for seed in args.seeds:
        success = run_curve_training(args.config_name, seed, args.base_output_dir)
        results[seed] = success

    # Print summary
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    for seed, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"Seed {seed:3d}: {status}")
    print("="*80 + "\n")

    # Return non-zero if any training failed
    if not all(results.values()):
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
