#!/usr/bin/env python3
"""Train a single shared-split VGG16/CIFAR10 pair with open_lth.

Workflow per run:
1. Train a shared trunk up to ``split_iter`` iterations.
2. Resume two branches from that exact checkpoint using different branch seeds.
3. Evaluate linear interpolation between the two final branch endpoints on train/test.

This uses open_lth's native CIFAR VGG16 with batch norm and default training recipe:
- SGD, lr=0.1, momentum=0.9, wd=1e-4
- training_steps=160ep
- milestone_steps=80ep,120ep
- CIFAR10 augmentations enabled by default
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OPEN_LTH_ROOT = PROJECT_ROOT / "external" / "open_lth"

sys.path.insert(0, str(OPEN_LTH_ROOT))

from datasets import registry as datasets_registry
from foundations import paths as open_lth_paths
from foundations.hparams import DatasetHparams, TrainingHparams
from foundations.step import Step
from models import registry as models_registry
from platforms import platform as platform_module
from platforms import local as local_platform
from training import checkpointing, standard_callbacks, train as train_lib
from training.metric_logger import MetricLogger
from training.optimizers import get_optimizer


@dataclass
class ScriptPlatform(local_platform.Platform):
    dataset_root_override: str = ""
    root_override: str = ""

    @property
    def dataset_root(self):
        return self.dataset_root_override or super().dataset_root

    @property
    def root(self):
        return self.root_override or super().root

    @property
    def imagenet_root(self):
        raise NotImplementedError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-split open_lth VGG16/CIFAR10 branch training.")
    parser.add_argument("--split-iter", type=int, required=True, help="Iteration at which to split.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/vgg16/cifar10/endpoints/open_lth_shared_split",
        help="Root output directory. Per-run results go under iter{split_iter}/.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(Path.home() / "open_lth_datasets"),
        help="Dataset root for open_lth CIFAR10.",
    )
    parser.add_argument("--shared-seed", type=int, default=42, help="Seed for shared trunk.")
    parser.add_argument("--branch-seeds", type=int, nargs=2, default=[0, 1], help="Two branch seeds.")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--num-eval-points", type=int, default=61, help="Interpolation evaluation points.")
    parser.add_argument(
        "--do-not-augment",
        action="store_true",
        help="Disable training augmentation. By default, open_lth CIFAR10 augmentation is used.",
    )
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_on_platform(platform: ScriptPlatform, fn):
    old_platform = platform_module._PLATFORM
    platform_module._PLATFORM = platform
    try:
        return fn()
    finally:
        platform_module._PLATFORM = old_platform


def get_default_hparams(batch_size: int, do_not_augment: bool) -> tuple:
    desc = models_registry.get_default_hparams("cifar_vgg_16")
    desc.dataset_hparams.batch_size = batch_size
    desc.dataset_hparams.do_not_augment = do_not_augment
    return desc.model_hparams, desc.dataset_hparams, desc.training_hparams


def build_model(model_hparams):
    return models_registry.get(model_hparams, outputs=10)


def copy_split_checkpoint(shared_dir: Path, branch_dir: Path) -> None:
    branch_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_dir / "checkpoint.pth", branch_dir / "checkpoint.pth")
    logger_path = shared_dir / "logger"
    if logger_path.exists():
        shutil.copy2(logger_path, branch_dir / "logger")


def save_initial_checkpoint(output_dir: Path, model, training_hparams: TrainingHparams, iterations_per_epoch: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = get_optimizer(training_hparams, model)
    logger = MetricLogger()
    step0 = Step.zero(iterations_per_epoch)
    checkpointing.save_checkpoint_callback(str(output_dir), step0, model, optimizer, logger)
    model.save(str(output_dir), step0)
    logger.save(str(output_dir))


def evaluate_model(model, loader) -> dict[str, float]:
    device = platform_module.get_platform().torch_device
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for examples, labels in loader:
            examples = examples.to(device)
            labels = labels.to(device)
            outputs = model(examples)
            loss = model.loss_criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

    return {"loss": total_loss / total, "accuracy": 100.0 * total_correct / total}


def evaluate_interpolation(model_hparams, dataset_hparams, state_a, state_b, num_points: int) -> dict:
    eval_hp = DatasetHparams(
        dataset_name=dataset_hparams.dataset_name,
        batch_size=dataset_hparams.batch_size,
        do_not_augment=True,
        transformation_seed=dataset_hparams.transformation_seed,
        subsample_fraction=dataset_hparams.subsample_fraction,
        random_labels_fraction=dataset_hparams.random_labels_fraction,
        unsupervised_labels=dataset_hparams.unsupervised_labels,
        blur_factor=dataset_hparams.blur_factor,
    )
    train_loader = datasets_registry.get(eval_hp, train=True)
    test_loader = datasets_registry.get(eval_hp, train=False)

    interp_model = build_model(model_hparams)
    ts = np.linspace(0.0, 1.0, num_points)
    results = {
        "t": ts.tolist(),
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    for t in ts:
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1.0 - t) * state_a[key] + t * state_b[key]
        interp_model.load_state_dict(interp_state)

        train_res = evaluate_model(interp_model, train_loader)
        test_res = evaluate_model(interp_model, test_loader)
        results["train_loss"].append(train_res["loss"])
        results["train_acc"].append(train_res["accuracy"])
        results["test_loss"].append(test_res["loss"])
        results["test_acc"].append(test_res["accuracy"])

    def loss_barrier(values):
        endpoint_avg = 0.5 * (values[0] + values[-1])
        return max(values) - endpoint_avg

    def acc_barrier(values):
        endpoint_avg = 0.5 * (values[0] + values[-1])
        return endpoint_avg - min(values)

    results["summary"] = {
        "train_loss_barrier": loss_barrier(results["train_loss"]),
        "test_loss_barrier": loss_barrier(results["test_loss"]),
        "train_acc_barrier": acc_barrier(results["train_acc"]),
        "test_acc_barrier": acc_barrier(results["test_acc"]),
        "min_train_acc": min(results["train_acc"]),
        "min_test_acc": min(results["test_acc"]),
        "max_train_loss": max(results["train_loss"]),
        "max_test_loss": max(results["test_loss"]),
    }
    return results


def write_interpolation_csv(path: Path, interpolation: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "train_loss", "train_acc", "test_loss", "test_acc"])
        for row in zip(
            interpolation["t"],
            interpolation["train_loss"],
            interpolation["train_acc"],
            interpolation["test_loss"],
            interpolation["test_acc"],
        ):
            writer.writerow(row)


def model_path_for_step(output_dir: Path, step: Step) -> Path:
    return Path(open_lth_paths.model(str(output_dir), step))


def main() -> None:
    args = parse_args()
    output_root = PROJECT_ROOT / args.output_root / f"iter{args.split_iter}"
    shared_dir = output_root / "shared"
    branch_dirs = [output_root / f"seed{seed}" for seed in args.branch_seeds]
    eval_dir = output_root / "evaluation"

    model_hparams, dataset_hparams, training_hparams = get_default_hparams(
        batch_size=args.batch_size,
        do_not_augment=args.do_not_augment,
    )

    platform = ScriptPlatform(
        num_workers=args.num_workers,
        dataset_root_override=args.dataset_root,
        root_override=str(PROJECT_ROOT / "external" / "open_lth_data"),
    )

    def job():
        iterations_per_epoch = datasets_registry.iterations_per_epoch(dataset_hparams)
        split_step = Step.from_iteration(args.split_iter, iterations_per_epoch)
        end_step = Step.from_str(training_hparams.training_steps, iterations_per_epoch)

        if output_root.exists():
            print(f"Removing existing output directory: {output_root}")
            shutil.rmtree(output_root)

        print("=" * 72)
        print("open_lth shared-split VGG16/CIFAR10 run")
        print("=" * 72)
        print(f"Output root: {output_root}")
        print(f"Dataset root: {args.dataset_root}")
        print(f"Split iteration: {args.split_iter}")
        print(f"Iterations/epoch: {iterations_per_epoch}")
        print(f"Training end step: {training_hparams.training_steps} -> ep={end_step.ep}, it={end_step.it}")
        print(f"Shared seed: {args.shared_seed}")
        print(f"Branch seeds: {args.branch_seeds}")
        print(f"Augmentation enabled: {not args.do_not_augment}")

        # Phase 1: shared trunk.
        print("\n[1/3] Training shared trunk")
        if args.split_iter == 0:
            set_global_seed(args.shared_seed)
            shared_model = build_model(model_hparams)
            save_initial_checkpoint(shared_dir, shared_model, training_hparams, iterations_per_epoch)
            print(f"Saved initial shared checkpoint to {shared_dir}")
        else:
            set_global_seed(args.shared_seed)
            shared_model = build_model(model_hparams)
            shared_train_hp = TrainingHparams(**vars(training_hparams))
            shared_train_hp.data_order_seed = args.shared_seed
            train_loader = datasets_registry.get(dataset_hparams, train=True)
            test_loader = datasets_registry.get(dataset_hparams, train=False)
            callbacks = [
                standard_callbacks.run_at_step(split_step, standard_callbacks.save_model),
                standard_callbacks.run_at_step(split_step, checkpointing.save_checkpoint_callback),
                standard_callbacks.run_at_step(split_step, standard_callbacks.save_logger),
                standard_callbacks.run_at_step(
                    split_step,
                    standard_callbacks.create_eval_callback("train", train_loader, verbose=True),
                ),
                standard_callbacks.run_at_step(
                    split_step,
                    standard_callbacks.create_eval_callback("test", test_loader, verbose=True),
                ),
                standard_callbacks.run_every_epoch(checkpointing.save_checkpoint_callback),
            ]
            train_lib.train(
                shared_train_hp,
                shared_model,
                train_loader,
                str(shared_dir),
                callbacks=callbacks,
                end_step=split_step,
            )
            print(f"Saved shared split checkpoint to {shared_dir / 'checkpoint.pth'}")

        # Phase 2: branches.
        print("\n[2/3] Training branch continuations")
        for branch_seed, branch_dir in zip(args.branch_seeds, branch_dirs):
            print(f"\nTraining branch seed {branch_seed} -> {branch_dir}")
            if branch_dir.exists():
                shutil.rmtree(branch_dir)
            copy_split_checkpoint(shared_dir, branch_dir)

            set_global_seed(branch_seed)
            branch_model = build_model(model_hparams)
            branch_train_hp = TrainingHparams(**vars(training_hparams))
            branch_train_hp.data_order_seed = branch_seed
            train_lib.standard_train(
                branch_model,
                str(branch_dir),
                dataset_hparams,
                branch_train_hp,
                verbose=True,
                evaluate_every_epoch=False,
            )

            final_model_path = model_path_for_step(branch_dir, end_step)
            print(f"Branch {branch_seed} final model: {final_model_path}")

        # Phase 3: interpolation eval.
        print("\n[3/3] Evaluating endpoint interpolation")
        state_paths = [model_path_for_step(branch_dir, end_step) for branch_dir in branch_dirs]
        state_a = torch.load(state_paths[0], map_location="cpu")
        state_b = torch.load(state_paths[1], map_location="cpu")
        interpolation = evaluate_interpolation(model_hparams, dataset_hparams, state_a, state_b, args.num_eval_points)

        eval_dir.mkdir(parents=True, exist_ok=True)
        interpolation_json = eval_dir / "interpolation.json"
        interpolation_csv = eval_dir / "interpolation.csv"
        with interpolation_json.open("w") as f:
            json.dump(
                {
                    "split_iter": args.split_iter,
                    "shared_seed": args.shared_seed,
                    "branch_seeds": args.branch_seeds,
                    "augmentation_enabled": not args.do_not_augment,
                    "endpoint_paths": [str(p.relative_to(PROJECT_ROOT)) for p in state_paths],
                    "interpolation": interpolation,
                },
                f,
                indent=2,
            )
        write_interpolation_csv(interpolation_csv, interpolation)
        print(f"Saved interpolation JSON to {interpolation_json}")
        print(f"Saved interpolation CSV to {interpolation_csv}")
        print(f"Interpolation summary: {json.dumps(interpolation['summary'], indent=2)}")

    run_on_platform(platform, job)


if __name__ == "__main__":
    main()
