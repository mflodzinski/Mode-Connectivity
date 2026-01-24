#!/usr/bin/env python3
"""
Verify the effect of shuffling on endpoint evaluation.

This script loads checkpoints and evaluates them on:
1. Original train/test splits
2. Shuffled train/test splits

This helps verify that the shuffle mechanism is working correctly.
"""

import argparse
import sys
import os
import torch
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external', 'dnn-mode-connectivity'))

import data
import models
import utils


def evaluate_checkpoint(checkpoint_path, dataset, data_path, model_name, transform_name,
                       use_shuffle=False, device='cuda'):
    """Evaluate a single checkpoint on both train and test sets."""

    print(f"\nEvaluating: {checkpoint_path}")
    print(f"  Shuffle: {use_shuffle}")

    # Load model architecture
    model_cfg = getattr(models, model_name)
    num_classes = 100 if dataset == 'CIFAR100' else 10
    model = model_cfg.base(num_classes=num_classes, **model_cfg.kwargs)
    model.to(device)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    # Load data (with or without shuffle)
    # Temporarily modify data.py behavior
    original_loaders_fn = data.loaders

    if use_shuffle:
        # Use shuffled data
        loaders_dict, num_classes = data.loaders(
            dataset=dataset,
            path=data_path,
            batch_size=128,
            num_workers=4,
            transform_name=transform_name,
            use_test=True,  # This triggers the shuffle
            shuffle_train=False  # Don't shuffle during loading
        )
    else:
        # Use original data - need to temporarily disable shuffle
        # We'll do this by creating a custom loader
        import torchvision
        import data as data_module

        ds = getattr(torchvision.datasets, dataset)
        path = os.path.join(data_path, dataset.lower())
        transform = getattr(getattr(data_module.Transforms, dataset), transform_name)

        train_set = ds(path, train=True, download=True, transform=transform.test)
        test_set = ds(path, train=False, download=True, transform=transform.test)

        loaders_dict = {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=128,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
        }

    # Update BN statistics on train set
    print("  Updating BN statistics...")
    utils.update_bn(loaders_dict['train'], model)

    # Evaluate on both splits
    criterion = torch.nn.CrossEntropyLoss()

    print("  Evaluating on train set...")
    train_res = utils.test(loaders_dict['train'], model, criterion)

    print("  Evaluating on test set...")
    test_res = utils.test(loaders_dict['test'], model, criterion)

    return {
        'train_loss': train_res['loss'],
        'train_error': 100.0 - train_res['accuracy'],
        'test_loss': test_res['loss'],
        'test_error': 100.0 - test_res['accuracy'],
    }


def main():
    parser = argparse.ArgumentParser(description='Verify shuffle effect on endpoint evaluation')
    parser.add_argument('--checkpoint0', type=str, required=True,
                       help='Path to first checkpoint (seed0)')
    parser.add_argument('--checkpoint1', type=str, required=True,
                       help='Path to second checkpoint (seed1)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (e.g., CIFAR10, FashionMNIST)')
    parser.add_argument('--data-path', type=str, default='./data',
                       help='Path to data directory')
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (e.g., VGG16, ConvFC)')
    parser.add_argument('--transform', type=str, default='VGG',
                       help='Transform name')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')

    args = parser.parse_args()

    print("=" * 80)
    print("ENDPOINT EVALUATION: Before and After Shuffle")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Transform: {args.transform}")

    # Evaluate both checkpoints with original split
    print("\n" + "=" * 80)
    print("ORIGINAL SPLIT (Standard Train/Test)")
    print("=" * 80)

    results_seed0_original = evaluate_checkpoint(
        args.checkpoint0, args.dataset, args.data_path,
        args.model, args.transform, use_shuffle=False, device=args.device
    )

    results_seed1_original = evaluate_checkpoint(
        args.checkpoint1, args.dataset, args.data_path,
        args.model, args.transform, use_shuffle=False, device=args.device
    )

    # Evaluate both checkpoints with shuffled split
    print("\n" + "=" * 80)
    print("SHUFFLED SPLIT (Combined + Shuffled + Split)")
    print("=" * 80)

    results_seed0_shuffled = evaluate_checkpoint(
        args.checkpoint0, args.dataset, args.data_path,
        args.model, args.transform, use_shuffle=True, device=args.device
    )

    results_seed1_shuffled = evaluate_checkpoint(
        args.checkpoint1, args.dataset, args.data_path,
        args.model, args.transform, use_shuffle=True, device=args.device
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nOriginal Split:")
    print(f"  Seed0 - Train: {results_seed0_original['train_error']:.2f}%, Test: {results_seed0_original['test_error']:.2f}%")
    print(f"  Seed1 - Train: {results_seed1_original['train_error']:.2f}%, Test: {results_seed1_original['test_error']:.2f}%")
    print(f"  Average Train Error: {(results_seed0_original['train_error'] + results_seed1_original['train_error'])/2:.2f}%")
    print(f"  Average Test Error:  {(results_seed0_original['test_error'] + results_seed1_original['test_error'])/2:.2f}%")

    print("\nShuffled Split:")
    print(f"  Seed0 - Train: {results_seed0_shuffled['train_error']:.2f}%, Test: {results_seed0_shuffled['test_error']:.2f}%")
    print(f"  Seed1 - Train: {results_seed1_shuffled['train_error']:.2f}%, Test: {results_seed1_shuffled['test_error']:.2f}%")
    print(f"  Average Train Error: {(results_seed0_shuffled['train_error'] + results_seed1_shuffled['train_error'])/2:.2f}%")
    print(f"  Average Test Error:  {(results_seed0_shuffled['test_error'] + results_seed1_shuffled['test_error'])/2:.2f}%")

    print("\nChange (Shuffled - Original):")
    print(f"  Train Error: {((results_seed0_shuffled['train_error'] + results_seed1_shuffled['train_error'])/2) - ((results_seed0_original['train_error'] + results_seed1_original['train_error'])/2):+.2f}%")
    print(f"  Test Error:  {((results_seed0_shuffled['test_error'] + results_seed1_shuffled['test_error'])/2) - ((results_seed0_original['test_error'] + results_seed1_original['test_error'])/2):+.2f}%")

    # Save results
    output_file = 'shuffle_verification_results.npz'
    np.savez(
        output_file,
        seed0_original=results_seed0_original,
        seed1_original=results_seed1_original,
        seed0_shuffled=results_seed0_shuffled,
        seed1_shuffled=results_seed1_shuffled
    )
    print(f"\nResults saved to: {output_file}")
    print("=" * 80)


if __name__ == '__main__':
    main()
