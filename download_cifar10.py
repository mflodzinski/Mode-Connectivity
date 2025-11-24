#!/usr/bin/env python3
"""Download CIFAR-10 dataset using torchvision."""

import torchvision
import torchvision.transforms as transforms

def download_cifar10(data_path='./data'):
    """Download CIFAR-10 train and test datasets."""
    print(f"Downloading CIFAR-10 to {data_path}...")

    # Download training set
    print("Downloading training set...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=True,
        download=True
    )
    print(f"Training set: {len(train_dataset)} images")

    # Download test set
    print("Downloading test set...")
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_path,
        train=False,
        download=True
    )
    print(f"Test set: {len(test_dataset)} images")

    print(f"\nCIFAR-10 downloaded successfully to {data_path}")
    print(f"Use this path in your config: {data_path}")

if __name__ == "__main__":
    download_cifar10()
