"""Model evaluation utilities.

Provides functions for evaluating models on datasets,
computing accuracy metrics, confusion matrices, etc.
Combines functionality from:
- scripts/analysis/lib/evaluation.py (single-model evaluation, metrics)
- scripts/eval/lib/evaluation.py (path evaluation along curves)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Any, Optional, Callable, List
from torch.utils.data import DataLoader


# ============================================================================
# Single-Model Evaluation (from analysis/lib/evaluation.py)
# ============================================================================

def evaluate_model(model: nn.Module,
                   loader: DataLoader,
                   device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                   return_features: bool = False,
                   return_logits: bool = True) -> Dict[str, np.ndarray]:
    """Evaluate model on dataset.

    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to run evaluation on
        return_features: Whether to return intermediate features
        return_logits: Whether to return logits

    Returns:
        Dictionary containing:
        - 'predictions': Predicted class indices
        - 'targets': Ground truth labels
        - 'correct': Boolean array of correct predictions
        - 'logits': Output logits (if return_logits=True)
        - 'features': Intermediate features (if return_features=True)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_correct = []
    all_logits = [] if return_logits else None
    all_features = [] if return_features else None

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            if return_features:
                # This assumes model has a features() method
                # May need customization per model
                features = model.features(inputs)
                logits = model.classifier(features)
                all_features.append(features.cpu().numpy())
            else:
                logits = model(inputs)

            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_correct.append((preds == targets).cpu().numpy())

            if return_logits:
                all_logits.append(logits.cpu().numpy())

    result = {
        'predictions': np.concatenate(all_preds),
        'targets': np.concatenate(all_targets),
        'correct': np.concatenate(all_correct),
    }

    if return_logits:
        result['logits'] = np.concatenate(all_logits)

    if return_features:
        result['features'] = np.concatenate(all_features)

    return result


def compute_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute overall accuracy.

    Args:
        predictions: Predicted class indices
        targets: Ground truth labels

    Returns:
        Accuracy as percentage (0-100)
    """
    return (predictions == targets).mean() * 100.0


def compute_per_class_accuracy(predictions: np.ndarray,
                               targets: np.ndarray,
                               num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-class accuracy.

    Args:
        predictions: Predicted class indices
        targets: Ground truth labels
        num_classes: Number of classes

    Returns:
        Tuple of (per_class_accuracies, per_class_counts)
        - per_class_accuracies: Accuracy for each class (percentage)
        - per_class_counts: Number of samples for each class
    """
    per_class_acc = []
    per_class_count = []

    for class_idx in range(num_classes):
        mask = targets == class_idx
        count = mask.sum()

        if count > 0:
            acc = (predictions[mask] == targets[mask]).mean() * 100.0
            per_class_acc.append(acc)
        else:
            per_class_acc.append(0.0)

        per_class_count.append(count)

    return np.array(per_class_acc), np.array(per_class_count)


def compute_confusion_matrix(predictions: np.ndarray,
                             targets: np.ndarray,
                             num_classes: int,
                             normalize: bool = False) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        predictions: Predicted class indices
        targets: Ground truth labels
        num_classes: Number of classes
        normalize: Whether to normalize by true class counts

    Returns:
        Confusion matrix of shape (num_classes, num_classes)
        Entry [i, j] is count (or percentage) of true class i predicted as class j
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)

    for i in range(len(predictions)):
        true_class = targets[i]
        pred_class = predictions[i]
        cm[true_class, pred_class] += 1

    if normalize:
        # Normalize by row (true class)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        cm = cm.astype(np.float32) / row_sums * 100.0

    return cm


def analyze_agreement(results1: Dict[str, np.ndarray],
                     results2: Dict[str, np.ndarray],
                     num_classes: int) -> Dict[str, Any]:
    """Analyze prediction agreement between two models.

    Args:
        results1: First model evaluation results (with 'predictions', 'targets', 'correct')
        results2: Second model evaluation results
        num_classes: Number of classes

    Returns:
        Dictionary containing:
        - 'agreement_rate': Overall prediction agreement percentage
        - 'both_correct': Percentage where both models correct
        - 'both_wrong': Percentage where both models wrong
        - 'only_model1_correct': Percentage where only model1 correct
        - 'only_model2_correct': Percentage where only model2 correct
        - 'per_class_agreement': Per-class agreement rates
    """
    preds1 = results1['predictions']
    preds2 = results2['predictions']
    targets = results1['targets']
    correct1 = results1['correct']
    correct2 = results2['correct']

    # Overall agreement
    agree = preds1 == preds2
    agreement_rate = agree.mean() * 100.0

    # Agreement patterns
    both_correct = (correct1 & correct2).mean() * 100.0
    both_wrong = (~correct1 & ~correct2).mean() * 100.0
    only_model1_correct = (correct1 & ~correct2).mean() * 100.0
    only_model2_correct = (~correct1 & correct2).mean() * 100.0

    # Per-class agreement
    per_class_agreement = []
    for class_idx in range(num_classes):
        mask = targets == class_idx
        if mask.sum() > 0:
            class_agreement = (preds1[mask] == preds2[mask]).mean() * 100.0
            per_class_agreement.append(class_agreement)
        else:
            per_class_agreement.append(0.0)

    return {
        'agreement_rate': agreement_rate,
        'both_correct': int(both_correct),
        'both_wrong': int(both_wrong),
        'only_model1_correct': int(only_model1_correct),
        'only_model2_correct': int(only_model2_correct),
        'per_class_agreement': np.array(per_class_agreement),
        'total_samples': len(targets),
    }


def compare_model_outputs(model1: nn.Module,
                         model2: nn.Module,
                         loader: DataLoader,
                         device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, Any]:
    """Compare outputs of two models on same dataset.

    Args:
        model1: First model
        model2: Second model
        loader: Data loader
        device: Device to run evaluation on

    Returns:
        Dictionary containing comparison metrics
    """
    results1 = evaluate_model(model1, loader, device)
    results2 = evaluate_model(model2, loader, device)

    num_classes = len(np.unique(results1['targets']))

    agreement = analyze_agreement(results1, results2, num_classes)

    # Max output difference
    max_logit_diff = np.abs(results1['logits'] - results2['logits']).max()

    return {
        'results1': results1,
        'results2': results2,
        'agreement': agreement,
        'max_logit_diff': max_logit_diff,
    }


# ============================================================================
# Path Evaluation (from eval/lib/evaluation.py)
# ============================================================================

class PathEvaluator:
    """Evaluate models along connectivity paths."""

    def __init__(self,
                 loaders: Dict,
                 device: torch.device,
                 criterion: Callable = F.cross_entropy):
        """Initialize path evaluator.

        Args:
            loaders: Dictionary with 'train' and 'test' DataLoaders
            device: Device to run evaluation on
            criterion: Loss criterion (default: cross-entropy)
        """
        self.loaders = loaders
        self.device = device
        self.criterion = criterion

    def evaluate_at_point(self,
                          model: nn.Module,
                          t_value: float,
                          update_bn: bool = True,
                          verbose: bool = False) -> Dict[str, float]:
        """Evaluate model at single point along path.

        Args:
            model: Model to evaluate
            t_value: Parameter value (for logging only)
            update_bn: Whether to update batch normalization statistics
            verbose: Print evaluation results

        Returns:
            Dictionary with keys: train_loss, train_acc, train_err,
                                 test_loss, test_acc, test_err
        """
        # Import utils from external repo
        import utils as external_utils

        # Update batch normalization statistics if requested
        if update_bn:
            external_utils.update_bn(self.loaders['train'], model, device=self.device)

        # Evaluate on train set
        train_res = external_utils.test(
            self.loaders['train'],
            model,
            self.criterion,
            device=self.device
        )

        # Evaluate on test set
        test_res = external_utils.test(
            self.loaders['test'],
            model,
            self.criterion,
            device=self.device
        )

        if verbose:
            print(f"t={t_value:.3f}: "
                  f"Train Loss={train_res['loss']:.4f}, Acc={train_res['accuracy']:.2f}% | "
                  f"Test Loss={test_res['loss']:.4f}, Acc={test_res['accuracy']:.2f}%")

        return {
            'train_loss': train_res['loss'],
            'train_acc': train_res['accuracy'],
            'train_err': 100.0 - train_res['accuracy'],
            'test_loss': test_res['loss'],
            'test_acc': test_res['accuracy'],
            'test_err': 100.0 - test_res['accuracy'],
        }

    def evaluate_path(self,
                      model: nn.Module,
                      ts: np.ndarray,
                      interpolator: Callable,
                      update_bn: bool = True,
                      verbose: bool = True,
                      **interp_kwargs) -> Dict[str, np.ndarray]:
        """Evaluate along full connectivity path.

        Args:
            model: Model to evaluate (weights will be updated at each t)
            ts: Array of t values to evaluate at
            interpolator: Function that takes (t, **interp_kwargs) and returns weights
            update_bn: Whether to update BN statistics at each point
            verbose: Print progress
            **interp_kwargs: Additional arguments passed to interpolator

        Returns:
            Dictionary with arrays: ts, tr_loss, tr_acc, tr_err, te_loss, te_acc, te_err
        """
        num_points = len(ts)

        # Initialize result arrays
        tr_loss = np.zeros(num_points)
        tr_acc = np.zeros(num_points)
        tr_err = np.zeros(num_points)
        te_loss = np.zeros(num_points)
        te_acc = np.zeros(num_points)
        te_err = np.zeros(num_points)

        if verbose:
            print(f"\nEvaluating path at {num_points} points...")
            print("=" * 70)

        # Evaluate at each point
        for i, t in enumerate(ts):
            # Get weights at t
            weights = interpolator(t, **interp_kwargs)

            # Load weights into model
            model.load_state_dict(weights)

            # Evaluate
            results = self.evaluate_at_point(model, t, update_bn=update_bn, verbose=verbose)

            # Store results
            tr_loss[i] = results['train_loss']
            tr_acc[i] = results['train_acc']
            tr_err[i] = results['train_err']
            te_loss[i] = results['test_loss']
            te_acc[i] = results['test_acc']
            te_err[i] = results['test_err']

        if verbose:
            print("=" * 70)
            print(f"✓ Evaluation complete")

        return {
            'ts': ts,
            'tr_loss': tr_loss,
            'tr_acc': tr_acc,
            'tr_err': tr_err,
            'te_loss': te_loss,
            'te_acc': te_acc,
            'te_err': te_err,
        }

    def evaluate_curve_path(self,
                           curve_model: nn.Module,
                           ts: np.ndarray,
                           update_bn: bool = True,
                           verbose: bool = True) -> Dict[str, np.ndarray]:
        """Evaluate curve model along path.

        Curve models handle interpolation internally via forward(x, t).

        Args:
            curve_model: CurveNet model
            ts: Array of t values to evaluate at
            update_bn: Whether to update BN statistics at each point
            verbose: Print progress

        Returns:
            Dictionary with arrays: ts, tr_loss, tr_acc, tr_err, te_loss, te_acc, te_err
        """
        # Import utils from external repo
        import utils as external_utils

        num_points = len(ts)

        # Initialize result arrays
        tr_loss = np.zeros(num_points)
        tr_acc = np.zeros(num_points)
        tr_err = np.zeros(num_points)
        te_loss = np.zeros(num_points)
        te_acc = np.zeros(num_points)
        te_err = np.zeros(num_points)

        if verbose:
            print(f"\nEvaluating curve at {num_points} points...")
            print("=" * 70)

        # Prepare t values as tensor
        coeffs_t = torch.FloatTensor([0.0]).to(self.device)

        # Evaluate at each point
        for i, t in enumerate(ts):
            coeffs_t[0] = t

            # Update batch normalization statistics
            if update_bn:
                external_utils.update_bn(
                    self.loaders['train'],
                    curve_model,
                    device=self.device,
                    **{'coeffs_t': coeffs_t}  # Pass t to curve model
                )

            # Evaluate on train set
            train_res = external_utils.test(
                self.loaders['train'],
                curve_model,
                self.criterion,
                device=self.device,
                **{'coeffs_t': coeffs_t}
            )

            # Evaluate on test set
            test_res = external_utils.test(
                self.loaders['test'],
                curve_model,
                self.criterion,
                device=self.device,
                **{'coeffs_t': coeffs_t}
            )

            # Store results
            tr_loss[i] = train_res['loss']
            tr_acc[i] = train_res['accuracy']
            tr_err[i] = 100.0 - train_res['accuracy']
            te_loss[i] = test_res['loss']
            te_acc[i] = test_res['accuracy']
            te_err[i] = 100.0 - test_res['accuracy']

            if verbose:
                print(f"t={t:.3f}: "
                      f"Train Loss={train_res['loss']:.4f}, Acc={train_res['accuracy']:.2f}% | "
                      f"Test Loss={test_res['loss']:.4f}, Acc={test_res['accuracy']:.2f}%")

        if verbose:
            print("=" * 70)
            print(f"✓ Curve evaluation complete")

        return {
            'ts': ts,
            'tr_loss': tr_loss,
            'tr_acc': tr_acc,
            'tr_err': tr_err,
            'te_loss': te_loss,
            'te_acc': te_acc,
            'te_err': te_err,
        }

    @staticmethod
    def compute_l2_norm(model: nn.Module) -> float:
        """Compute L2 norm of model parameters.

        Args:
            model: Model to compute norm for

        Returns:
            L2 norm (scalar)
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total_norm += torch.sum(param.data ** 2).item()
        return total_norm ** 0.5
