"""
Argument parser utilities.

Provides reusable argument parser components for common arguments.
"""

import argparse
from typing import Optional


class ArgumentParserBuilder:
    """Helper class to build argument parsers with common argument groups."""

    @staticmethod
    def add_checkpoint_args(parser: argparse.ArgumentParser,
                           single: bool = False,
                           required: bool = True) -> None:
        """
        Add checkpoint-related arguments.

        Args:
            parser: ArgumentParser to add arguments to
            single: If True, add single --checkpoint arg; if False, add --checkpoint1/2
            required: Whether checkpoint arguments are required
        """
        if single:
            parser.add_argument('--checkpoint', type=str, required=required,
                              help='Path to checkpoint file')
        else:
            parser.add_argument('--checkpoint1', type=str, required=required,
                              help='Path to first checkpoint file')
            parser.add_argument('--checkpoint2', type=str, required=required,
                              help='Path to second checkpoint file')

    @staticmethod
    def add_model_args(parser: argparse.ArgumentParser,
                      default_model: str = 'VGG16',
                      default_num_classes: int = 10) -> None:
        """
        Add model architecture arguments.

        Args:
            parser: ArgumentParser to add arguments to
            default_model: Default model name
            default_num_classes: Default number of classes
        """
        parser.add_argument('--model', type=str, default=default_model,
                          help='Model architecture (e.g., VGG16, VGG19, ResNet18)')
        parser.add_argument('--num-classes', type=int, default=default_num_classes,
                          help='Number of output classes')
        parser.add_argument('--use-bn', action='store_true', default=False,
                          help='Use batch normalization variant of model')

    @staticmethod
    def add_dataset_args(parser: argparse.ArgumentParser,
                        default_dataset: str = 'CIFAR10',
                        default_batch_size: int = 128) -> None:
        """
        Add dataset-related arguments.

        Args:
            parser: ArgumentParser to add arguments to
            default_dataset: Default dataset name
            default_batch_size: Default batch size
        """
        parser.add_argument('--dataset', type=str, default=default_dataset,
                          help='Dataset name (e.g., CIFAR10, CIFAR100)')
        parser.add_argument('--data-path', type=str, default='./data',
                          help='Path to dataset directory')
        parser.add_argument('--transform', type=str, default='VGG',
                          help='Transform/augmentation to use')
        parser.add_argument('--batch-size', type=int, default=default_batch_size,
                          help='Batch size for data loading')
        parser.add_argument('--num-workers', type=int, default=4,
                          help='Number of data loading workers')

    @staticmethod
    def add_output_args(parser: argparse.ArgumentParser,
                       single_file: bool = True,
                       required: bool = True) -> None:
        """
        Add output-related arguments.

        Args:
            parser: ArgumentParser to add arguments to
            single_file: If True, add --output; if False, add --output-dir
            required: Whether output argument is required
        """
        if single_file:
            parser.add_argument('--output', type=str, required=required,
                              help='Output file path')
        else:
            parser.add_argument('--output-dir', type=str, required=required,
                              help='Output directory path')

    @staticmethod
    def add_device_args(parser: argparse.ArgumentParser) -> None:
        """
        Add device/CUDA arguments.

        Args:
            parser: ArgumentParser to add arguments to
        """
        parser.add_argument('--device', type=str, default='cuda',
                          help='Device to use (cuda or cpu)')
        parser.add_argument('--cuda', action='store_true', default=False,
                          help='Use CUDA if available')

    @staticmethod
    def add_curve_args(parser: argparse.ArgumentParser,
                      required: bool = True) -> None:
        """
        Add curve-related arguments.

        Args:
            parser: ArgumentParser to add arguments to
            required: Whether curve checkpoint is required
        """
        parser.add_argument('--curve', type=str, required=required,
                          help='Path to curve checkpoint file')
        parser.add_argument('--num-bends', type=int, default=3,
                          help='Number of bend points in curve')

    @staticmethod
    def add_plot_output_args(parser: argparse.ArgumentParser,
                            required: bool = True) -> None:
        """
        Add plotting output arguments.

        Args:
            parser: ArgumentParser to add arguments to
            required: Whether --output is required
        """
        parser.add_argument('--output', type=str, required=required,
                          help='Output path for plot/figure')
        parser.add_argument('--show', action='store_true',
                          help='Display plot interactively')

    @staticmethod
    def add_dimred_args(parser: argparse.ArgumentParser) -> None:
        """
        Add dimensionality reduction arguments.

        Args:
            parser: ArgumentParser to add arguments to
        """
        parser.add_argument('--method', type=str, default='umap',
                          choices=['umap', 'pca'],
                          help='Dimensionality reduction method (default: umap)')
        parser.add_argument('--random-state', type=int, default=42,
                          help='Random seed for reproducibility (default: 42)')

    @staticmethod
    def add_animation_args(parser: argparse.ArgumentParser) -> None:
        """
        Add animation-related arguments.

        Args:
            parser: ArgumentParser to add arguments to
        """
        parser.add_argument('--fps', type=int, default=5,
                          help='Frames per second for animation (default: 5)')
        parser.add_argument('--skip-frames', type=int, default=1,
                          help='Only render every Nth frame (default: 1)')

    @staticmethod
    def create_basic_parser(description: str,
                          add_checkpoints: Optional[str] = None,
                          add_model: bool = False,
                          add_dataset: bool = False,
                          add_output: Optional[str] = None) -> argparse.ArgumentParser:
        """
        Create a basic parser with commonly used arguments.

        Args:
            description: Parser description
            add_checkpoints: 'single', 'dual', or None
            add_model: Whether to add model arguments
            add_dataset: Whether to add dataset arguments
            add_output: 'file', 'dir', or None

        Returns:
            Configured ArgumentParser
        """
        parser = argparse.ArgumentParser(description=description)

        if add_checkpoints == 'single':
            ArgumentParserBuilder.add_checkpoint_args(parser, single=True)
        elif add_checkpoints == 'dual':
            ArgumentParserBuilder.add_checkpoint_args(parser, single=False)

        if add_model:
            ArgumentParserBuilder.add_model_args(parser)

        if add_dataset:
            ArgumentParserBuilder.add_dataset_args(parser)

        if add_output == 'file':
            ArgumentParserBuilder.add_output_args(parser, single_file=True)
        elif add_output == 'dir':
            ArgumentParserBuilder.add_output_args(parser, single_file=False)

        return parser
