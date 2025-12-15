"""Dataset loading and preprocessing utilities.

Provides dataset loaders and CIFAR-10 specific utilities.
Unified from scripts/analysis/lib/data.py and scripts/eval/lib/setup.py
"""

import sys
import numpy as np
from typing import Dict, Tuple, List

# Add external dependencies to path
sys.path.insert(0, 'external/dnn-mode-connectivity')
import data as dnn_data


class CIFAR10Utils:
    """CIFAR-10 specific constants and utilities."""

    CLASS_NAMES = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]

    MEAN = np.array([0.4914, 0.4822, 0.4465])
    STD = np.array([0.2470, 0.2435, 0.2616])

    @classmethod
    def get_class_name(cls, idx: int) -> str:
        """Get class name for given index."""
        return cls.CLASS_NAMES[idx]

    @classmethod
    def get_class_names(cls) -> List[str]:
        """Get all class names."""
        return cls.CLASS_NAMES.copy()

    @classmethod
    def denormalize(cls, img: np.ndarray) -> np.ndarray:
        """Denormalize CIFAR-10 image to displayable format.

        Args:
            img: Normalized image in CHW format (C, H, W)

        Returns:
            Denormalized image in HWC format (H, W, C) with values in [0, 255]
        """
        # Transpose from CHW to HWC
        img = img.transpose(1, 2, 0)

        # Denormalize
        img = img * cls.STD + cls.MEAN

        # Clip and convert to uint8
        img = np.clip(img * 255, 0, 255).astype(np.uint8)

        return img


def get_loaders(dataset: str,
                data_path: str = './data',
                batch_size: int = 128,
                num_workers: int = 4,
                transform_name: str = 'VGG',
                use_test: bool = True,
                shuffle_train: bool = True) -> Tuple[Dict, int]:
    """Get data loaders for specified dataset.

    Args:
        dataset: Dataset name (e.g., 'CIFAR10', 'CIFAR100')
        data_path: Path to dataset directory
        batch_size: Batch size for loaders
        num_workers: Number of worker processes
        transform_name: Transform to apply (e.g., 'VGG')
        use_test: Whether to use test set (vs validation)
        shuffle_train: Whether to shuffle training data

    Returns:
        Tuple of (loaders_dict, num_classes)
        loaders_dict contains keys: 'train', 'test' (or 'val')
    """
    loaders, num_classes = dnn_data.loaders(
        dataset,
        path=data_path,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_name=transform_name,
        use_test=use_test,
        shuffle_train=shuffle_train
    )

    return loaders, num_classes


def get_class_names(dataset: str) -> List[str]:
    """Get class names for dataset.

    Args:
        dataset: Dataset name

    Returns:
        List of class names
    """
    if dataset.upper() == 'CIFAR10':
        return CIFAR10Utils.get_class_names()
    else:
        # For other datasets, return numeric labels
        raise NotImplementedError(f"Class names not implemented for {dataset}")
