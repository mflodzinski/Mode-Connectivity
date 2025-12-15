"""File I/O and result saving utilities.

Provides functions for saving/loading JSON, NPZ files,
and creating directories. Combines functionality from:
- scripts/analysis/lib/io.py
- scripts/eval/lib/output.py
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Union, Optional


# ============================================================================
# Basic I/O (from analysis/lib/io.py)
# ============================================================================

def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file.

    Args:
        data: Data to save (must be JSON serializable)
        path: Output file path
        indent: Indentation level for pretty printing
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)


def load_json(path: Union[str, Path]) -> Any:
    """Load data from JSON file.

    Args:
        path: Input file path

    Returns:
        Loaded data
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_npz(path: Union[str, Path], **arrays) -> None:
    """Save arrays to NPZ file.

    Args:
        path: Output file path
        **arrays: Named arrays to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(path, **arrays)


def load_npz(path: Union[str, Path]) -> Dict[str, np.ndarray]:
    """Load arrays from NPZ file.

    Args:
        path: Input file path

    Returns:
        Dictionary of loaded arrays
    """
    data = np.load(path)
    return {key: data[key] for key in data.files}


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating if necessary.

    Args:
        path: Directory path

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(file_path: Union[str, Path]) -> Path:
    """Ensure parent directory of file exists.

    Args:
        file_path: File path

    Returns:
        Path object for the file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def save_text(content: str, path: Union[str, Path]) -> None:
    """Save text content to file.

    Args:
        content: Text content to save
        path: Output file path
    """
    path = ensure_parent_dir(path)

    with open(path, 'w') as f:
        f.write(content)


def load_text(path: Union[str, Path]) -> str:
    """Load text content from file.

    Args:
        path: Input file path

    Returns:
        Text content
    """
    with open(path, 'r') as f:
        return f.read()


# ============================================================================
# Specialized Result Savers (from eval/lib/output.py)
# ============================================================================

class ResultSaver:
    """Save evaluation results to NPZ files with standardized formats."""

    @staticmethod
    def save_standard(output_path: str,
                      ts: np.ndarray,
                      train_metrics: Dict[str, np.ndarray],
                      test_metrics: Dict[str, np.ndarray],
                      **extra) -> None:
        """Save standard evaluation results.

        Standard format includes:
        - ts: t values
        - tr_loss, tr_acc, tr_err: training metrics
        - te_loss, te_acc, te_err: test metrics

        Args:
            output_path: Path to save .npz file
            ts: Array of t values
            train_metrics: Dict with keys 'loss', 'acc', 'err'
            test_metrics: Dict with keys 'loss', 'acc', 'err'
            **extra: Additional arrays to save
        """
        np.savez(
            output_path,
            ts=ts,
            tr_loss=train_metrics['loss'],
            tr_acc=train_metrics['acc'],
            tr_err=train_metrics['err'],
            te_loss=test_metrics['loss'],
            te_acc=test_metrics['acc'],
            te_err=test_metrics['err'],
            **extra
        )

    @staticmethod
    def save_from_dict(output_path: str, results: Dict[str, np.ndarray]) -> None:
        """Save results dictionary directly to NPZ.

        Args:
            output_path: Path to save .npz file
            results: Dictionary of arrays to save
        """
        np.savez(output_path, **results)

    @staticmethod
    def save_with_l2(output_path: str,
                     ts: np.ndarray,
                     train_metrics: Dict[str, np.ndarray],
                     test_metrics: Dict[str, np.ndarray],
                     l2_norm: np.ndarray) -> None:
        """Save results including L2 norms along path.

        Used for linear interpolation evaluation.

        Args:
            output_path: Path to save .npz file
            ts: Array of t values
            train_metrics: Dict with keys 'loss', 'acc', 'err'
            test_metrics: Dict with keys 'loss', 'acc', 'err'
            l2_norm: Array of L2 norms at each t
        """
        np.savez(
            output_path,
            ts=ts,
            tr_loss=train_metrics['loss'],
            tr_acc=train_metrics['acc'],
            tr_err=train_metrics['err'],
            te_loss=test_metrics['loss'],
            te_acc=test_metrics['acc'],
            te_err=test_metrics['err'],
            l2_norm=l2_norm
        )

    @staticmethod
    def save_comparison(output_path: str,
                       ts: np.ndarray,
                       linear_results: Dict[str, np.ndarray],
                       symplane_results: Dict[str, np.ndarray]) -> None:
        """Save comparison results (linear vs symmetry plane).

        Args:
            output_path: Path to save .npz file
            ts: Array of t values
            linear_results: Linear path results with keys: tr_loss, tr_acc, te_loss, te_acc
            symplane_results: Symmetry plane results with same keys
        """
        np.savez(
            output_path,
            ts=ts,
            # Linear path results
            linear_tr_loss=linear_results['tr_loss'],
            linear_tr_acc=linear_results['tr_acc'],
            linear_tr_err=linear_results['tr_err'],
            linear_te_loss=linear_results['te_loss'],
            linear_te_acc=linear_results['te_acc'],
            linear_te_err=linear_results['te_err'],
            # Symmetry plane results
            symplane_tr_loss=symplane_results['tr_loss'],
            symplane_tr_acc=symplane_results['tr_acc'],
            symplane_tr_err=symplane_results['tr_err'],
            symplane_te_loss=symplane_results['te_loss'],
            symplane_te_acc=symplane_results['te_acc'],
            symplane_te_err=symplane_results['te_err'],
        )

    @staticmethod
    def save_detailed_predictions(output_path: str,
                                  predictions: np.ndarray,
                                  targets: np.ndarray,
                                  features_t0: np.ndarray,
                                  features_t1: np.ndarray,
                                  images: np.ndarray,
                                  ts: np.ndarray) -> None:
        """Save detailed per-sample predictions.

        Used for detailed curve analysis with feature extraction.

        Args:
            output_path: Path to save .npz file
            predictions: Array of shape (num_samples, num_t_values, num_classes)
            targets: Array of ground truth labels (num_samples,)
            features_t0: Features from endpoint 0 model (num_samples, feature_dim)
            features_t1: Features from endpoint 1 model (num_samples, feature_dim)
            images: Array of sample images (num_samples, C, H, W)
            ts: Array of t values used for predictions
        """
        np.savez(
            output_path,
            predictions=predictions,
            targets=targets,
            features_t0=features_t0,
            features_t1=features_t1,
            images=images,
            ts=ts
        )

    @staticmethod
    def ensure_output_dir(output_path: str) -> str:
        """Ensure output directory exists.

        Args:
            output_path: Path to output file

        Returns:
            Absolute path with ensured directory
        """
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return os.path.abspath(output_path)

    @staticmethod
    def get_standard_output_path(output_dir: str, filename: str) -> str:
        """Get standard output path and ensure directory exists.

        Args:
            output_dir: Output directory
            filename: Output filename (e.g., 'linear.npz', 'curve.npz')

        Returns:
            Full output path with ensured directory
        """
        output_path = os.path.join(output_dir, filename)
        return ResultSaver.ensure_output_dir(output_path)
