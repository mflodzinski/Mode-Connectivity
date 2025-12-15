"""
Prediction analysis utilities for mode connectivity curves.

Provides utilities for analyzing prediction changes, comparing endpoints,
and identifying unstable samples along curves.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from PIL import Image

from ..core import data as lib_data, output as io


class PredictionAnalyzer:
    """Analyze prediction changes and patterns along curves."""

    def __init__(self, predictions_file: Optional[str] = None):
        """Initialize PredictionAnalyzer.

        Args:
            predictions_file: Path to predictions NPZ file (optional)
        """
        self.predictions_file = predictions_file
        self.data = None
        self.class_names = lib_data.CIFAR10Utils.CLASS_NAMES

    def load_predictions(self) -> Dict[str, np.ndarray]:
        """Load predictions from NPZ file.

        Returns:
            Dictionary with predictions data
        """
        if self.predictions_file is None:
            raise ValueError("No predictions file specified")

        self.data = io.load_npz(self.predictions_file)
        return self.data

    def analyze_changes(self, verbose: bool = True) -> Dict[str, Any]:
        """Identify samples that change predictions along curve.

        Args:
            verbose: Whether to print analysis details

        Returns:
            Dictionary with analysis results
        """
        if self.data is None:
            self.load_predictions()

        predictions = self.data['predictions']  # [num_points, num_samples]
        targets = self.data['targets']  # [num_samples]

        num_points, num_samples = predictions.shape

        if verbose:
            print(f"\nAnalyzing predictions for {num_samples} samples across {num_points} points...")

        # Find samples that change predictions
        initial_pred = predictions[0]  # Predictions at t=0
        changes_mask = np.zeros(num_samples, dtype=bool)

        for i in range(num_samples):
            # Check if prediction ever differs from initial
            if not np.all(predictions[:, i] == initial_pred[i]):
                changes_mask[i] = True

        changing_indices = np.where(changes_mask)[0]
        stable_indices = np.where(~changes_mask)[0]

        num_changing = len(changing_indices)
        num_stable = len(stable_indices)

        if verbose:
            print(f"\nPrediction stability:")
            print(f"  Samples that change: {num_changing} ({num_changing/num_samples*100:.2f}%)")
            print(f"  Samples that stay stable: {num_stable} ({num_stable/num_samples*100:.2f}%)")

        # Count prediction changes for each sample
        change_counts = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            pred_trajectory = predictions[:, i]
            # Count transitions
            changes = np.sum(pred_trajectory[:-1] != pred_trajectory[1:])
            change_counts[i] = changes

        # Statistics on changing samples
        if num_changing > 0 and verbose:
            changing_counts = change_counts[changing_indices]
            print(f"\nChange count statistics (for changing samples):")
            print(f"  Min changes: {changing_counts.min()}")
            print(f"  Max changes: {changing_counts.max()}")
            print(f"  Mean changes: {changing_counts.mean():.2f}")
            print(f"  Median changes: {np.median(changing_counts):.1f}")

            # Find most unstable sample
            most_unstable_idx = changing_indices[np.argmax(changing_counts)]
            print(f"\nMost unstable sample:")
            print(f"  Index: {most_unstable_idx}")
            print(f"  Number of changes: {change_counts[most_unstable_idx]}")
            print(f"  True label: {targets[most_unstable_idx]}")

        # Per-class analysis
        if verbose:
            print(f"\nPer-class stability:")
            for class_idx in range(10):
                class_mask = targets == class_idx
                class_samples = np.sum(class_mask)
                class_changing = np.sum(changes_mask & class_mask)

                if class_samples > 0:
                    pct_changing = class_changing / class_samples * 100
                    print(f"  {self.class_names[class_idx]:<12}: {class_changing:4d}/{class_samples:4d} "
                          f"({pct_changing:5.2f}%) change predictions")

        # Correctness analysis
        initial_correct = predictions[0] == targets
        final_correct = predictions[-1] == targets

        both_correct = np.sum(initial_correct & final_correct)
        both_wrong = np.sum(~initial_correct & ~final_correct)
        correct_to_wrong = np.sum(initial_correct & ~final_correct)
        wrong_to_correct = np.sum(~initial_correct & final_correct)

        if verbose:
            print(f"\nEndpoint correctness:")
            print(f"  Both endpoints correct: {both_correct} ({both_correct/num_samples*100:.2f}%)")
            print(f"  Both endpoints wrong: {both_wrong} ({both_wrong/num_samples*100:.2f}%)")
            print(f"  Correct→Wrong: {correct_to_wrong} ({correct_to_wrong/num_samples*100:.2f}%)")
            print(f"  Wrong→Correct: {wrong_to_correct} ({wrong_to_correct/num_samples*100:.2f}%)")

        return {
            'num_samples': num_samples,
            'num_changing': num_changing,
            'num_stable': num_stable,
            'changing_indices': changing_indices,
            'stable_indices': stable_indices,
            'change_counts': change_counts,
            'both_correct': both_correct,
            'both_wrong': both_wrong,
            'correct_to_wrong': correct_to_wrong,
            'wrong_to_correct': wrong_to_correct
        }

    def compare_endpoints(self, verbose: bool = True) -> Dict[str, Any]:
        """Compare per-class accuracy between endpoints.

        Args:
            verbose: Whether to print comparison details

        Returns:
            Dictionary with comparison statistics
        """
        if self.data is None:
            self.load_predictions()

        predictions = self.data['predictions']
        targets = self.data['targets']

        # Get endpoint predictions
        pred_t0 = predictions[0]
        pred_t1 = predictions[-1]

        # Compute per-class accuracy at each endpoint
        per_class_acc_t0 = []
        per_class_acc_t1 = []
        per_class_counts = []

        for class_idx in range(10):
            class_mask = targets == class_idx
            count = class_mask.sum()

            if count > 0:
                acc_t0 = (pred_t0[class_mask] == targets[class_mask]).mean() * 100
                acc_t1 = (pred_t1[class_mask] == targets[class_mask]).mean() * 100
            else:
                acc_t0 = 0.0
                acc_t1 = 0.0

            per_class_acc_t0.append(acc_t0)
            per_class_acc_t1.append(acc_t1)
            per_class_counts.append(count)

        per_class_acc_t0 = np.array(per_class_acc_t0)
        per_class_acc_t1 = np.array(per_class_acc_t1)
        per_class_counts = np.array(per_class_counts)

        # Compute differences
        differences = per_class_acc_t1 - per_class_acc_t0

        if verbose:
            print(f"\n{'='*70}")
            print("PER-CLASS ACCURACY COMPARISON")
            print(f"{'='*70}")
            print(f"\n{'Class':<12} {'t=0':<10} {'t=1':<10} {'Diff':<10} {'Count':<8}")
            print("-" * 70)

            for i, name in enumerate(self.class_names):
                print(f"{name:<12} {per_class_acc_t0[i]:>9.2f}% {per_class_acc_t1[i]:>9.2f}% "
                      f"{differences[i]:>+9.2f}% {per_class_counts[i]:>8}")

            print("-" * 70)
            overall_t0 = (pred_t0 == targets).mean() * 100
            overall_t1 = (pred_t1 == targets).mean() * 100
            print(f"{'Overall':<12} {overall_t0:>9.2f}% {overall_t1:>9.2f}% "
                  f"{overall_t1 - overall_t0:>+9.2f}%")
            print("=" * 70)

        return {
            'per_class_acc_t0': per_class_acc_t0,
            'per_class_acc_t1': per_class_acc_t1,
            'per_class_counts': per_class_counts,
            'differences': differences,
            'overall_acc_t0': (pred_t0 == targets).mean() * 100,
            'overall_acc_t1': (pred_t1 == targets).mean() * 100
        }

    def find_unstable_samples(self, top_k: int = 20) -> np.ndarray:
        """Find samples with most prediction changes.

        Args:
            top_k: Number of top unstable samples to return

        Returns:
            Array of sample indices sorted by instability
        """
        if self.data is None:
            self.load_predictions()

        predictions = self.data['predictions']
        num_samples = predictions.shape[1]

        # Count changes for each sample
        change_counts = np.zeros(num_samples, dtype=int)

        for i in range(num_samples):
            pred_trajectory = predictions[:, i]
            changes = np.sum(pred_trajectory[:-1] != pred_trajectory[1:])
            change_counts[i] = changes

        # Get top k unstable samples
        top_indices = np.argsort(change_counts)[::-1][:top_k]

        return top_indices

    def save_sample_images(self,
                          output_dir: str,
                          dataset_path: str = './data',
                          top_k: int = 20):
        """Save images of most unstable samples.

        Args:
            output_dir: Directory to save images
            dataset_path: Path to dataset
            top_k: Number of top unstable samples to save
        """
        from torch.utils.data import DataLoader
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms

        if self.data is None:
            self.load_predictions()

        output_path = io.ensure_dir(output_dir)

        # Find unstable samples
        unstable_indices = self.find_unstable_samples(top_k)

        targets = self.data['targets']
        predictions = self.data['predictions']

        # Load CIFAR-10 test set
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        testset = datasets.CIFAR10(root=dataset_path, train=False,
                                   download=False, transform=transform)

        print(f"\nSaving top {top_k} unstable sample images to {output_dir}/")

        for rank, idx in enumerate(unstable_indices):
            # Get image
            img_tensor, true_label = testset[idx]

            # Denormalize
            img_array = img_tensor.numpy()
            img_denorm = lib_data.CIFAR10Utils.denormalize(img_array)

            # Count prediction changes
            pred_trajectory = predictions[:, idx]
            num_changes = np.sum(pred_trajectory[:-1] != pred_trajectory[1:])

            # Create filename with metadata
            true_class = self.class_names[true_label]
            pred_t0 = predictions[0, idx]
            pred_t1 = predictions[-1, idx]

            filename = (f"rank{rank+1:02d}_idx{idx:05d}_"
                       f"true{true_class}_"
                       f"pred{self.class_names[pred_t0]}_to_{self.class_names[pred_t1]}_"
                       f"changes{num_changes}.png")

            # Save image (upscaled for visibility)
            img_pil = Image.fromarray(img_denorm)
            img_pil = img_pil.resize((256, 256), Image.NEAREST)
            img_pil.save(output_path / filename)

            print(f"  {rank+1:2d}. Sample {idx:5d}: {num_changes} changes "
                  f"({true_class} → {self.class_names[pred_t0]} → {self.class_names[pred_t1]})")

        print(f"\n✓ Saved {top_k} images to {output_dir}/")

    def generate_report(self, output_dir: str, verbose: bool = True):
        """Generate comprehensive analysis report.

        Args:
            output_dir: Directory to save report
            verbose: Whether to print progress
        """
        output_path = io.ensure_dir(output_dir)

        if verbose:
            print(f"\n{'='*70}")
            print("GENERATING PREDICTION ANALYSIS REPORT")
            print(f"{'='*70}")

        # Run analyses
        changes = self.analyze_changes(verbose=verbose)
        endpoints = self.compare_endpoints(verbose=verbose)

        # Save results
        results = {
            'prediction_changes': {
                'num_samples': int(changes['num_samples']),
                'num_changing': int(changes['num_changing']),
                'num_stable': int(changes['num_stable']),
                'both_correct': int(changes['both_correct']),
                'both_wrong': int(changes['both_wrong']),
                'correct_to_wrong': int(changes['correct_to_wrong']),
                'wrong_to_correct': int(changes['wrong_to_correct'])
            },
            'endpoint_comparison': {
                'per_class_acc_t0': endpoints['per_class_acc_t0'].tolist(),
                'per_class_acc_t1': endpoints['per_class_acc_t1'].tolist(),
                'differences': endpoints['differences'].tolist(),
                'class_names': self.class_names,
                'overall_acc_t0': float(endpoints['overall_acc_t0']),
                'overall_acc_t1': float(endpoints['overall_acc_t1'])
            }
        }

        # Save JSON report
        io.save_json(results, output_path / 'prediction_analysis.json')

        # Save NPZ with detailed data
        io.save_npz(
            output_path / 'prediction_analysis_detailed.npz',
            changing_indices=changes['changing_indices'],
            stable_indices=changes['stable_indices'],
            change_counts=changes['change_counts']
        )

        if verbose:
            print(f"\n✓ Report saved to {output_dir}/")
            print("=" * 70)
