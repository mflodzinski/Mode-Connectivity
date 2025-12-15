"""
Curve analysis utilities for mode connectivity.

Provides utilities for analyzing Bezier curves, extracting bend points,
verifying symmetry, and computing distances along curves.
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Add external dependencies
sys.path.insert(0, 'external/dnn-mode-connectivity')
import curves as dnn_curves

from ..core import checkpoint
from ..evaluation import metrics


class CurveAnalyzer:
    """Analyze Bezier curves and their properties."""

    def __init__(self, curve_checkpoint: Optional[str] = None):
        """Initialize CurveAnalyzer.

        Args:
            curve_checkpoint: Path to curve checkpoint file (optional)
        """
        self.curve_checkpoint = curve_checkpoint
        self.curve_state = None
        self.bends = []
        self.num_bends = 3  # Default for Bezier curves

    def load_curve(self) -> Dict[str, torch.Tensor]:
        """Load curve model from checkpoint.

        Returns:
            Curve state dictionary
        """
        if self.curve_checkpoint is None:
            raise ValueError("No curve checkpoint specified")

        self.curve_state = checkpoint.load_state_dict(self.curve_checkpoint)
        return self.curve_state

    def extract_bends(self, num_bends: int = 3) -> List[Dict[str, torch.Tensor]]:
        """Extract bend point parameters from curve state dict.

        Args:
            num_bends: Number of bend points (default: 3)

        Returns:
            List of state dicts, one per bend point
        """
        if self.curve_state is None:
            self.load_curve()

        self.num_bends = num_bends
        self.bends = []

        for i in range(num_bends):
            bend_state = {}
            for key in self.curve_state.keys():
                if key.endswith(f'.{i}'):
                    # Remove the bend index suffix to get parameter name
                    param_name = key[:-2]  # Remove '.i'
                    bend_state[param_name] = self.curve_state[key]
            self.bends.append(bend_state)

        return self.bends

    def get_params_at_t(self, t: float, num_bends: int = 3) -> Dict[str, torch.Tensor]:
        """Extract parameter values at specific t value using Bezier interpolation.

        Args:
            t: Parameter value (typically in [0, 1])
            num_bends: Number of bend points

        Returns:
            State dict with interpolated weights
        """
        from scipy.special import comb

        if not self.bends:
            self.extract_bends(num_bends)

        if len(self.bends) == 0:
            raise ValueError("No bend points found in curve state")

        # Initialize result
        result = {}

        # Get parameter names from first bend
        param_names = self.bends[0].keys()

        # Bezier interpolation
        n = num_bends - 1  # degree of curve

        for param_name in param_names:
            weighted_sum = None

            for i, bend in enumerate(self.bends):
                # Bezier basis function
                coeff = comb(n, i, exact=True) * ((1 - t) ** (n - i)) * (t ** i)

                if weighted_sum is None:
                    weighted_sum = coeff * bend[param_name]
                else:
                    weighted_sum = weighted_sum + coeff * bend[param_name]

            result[param_name] = weighted_sum

        return result

    def compute_layer_distances_along_curve(self,
                                           original_checkpoint: str,
                                           num_points: int = 61,
                                           permutation_invariant: bool = True) -> Dict[str, Any]:
        """Analyze layer-wise distances along the curve.

        Args:
            original_checkpoint: Path to original model checkpoint
            num_points: Number of points to sample along curve
            permutation_invariant: Use permutation-invariant distance metric

        Returns:
            Dictionary with distance statistics for each layer at each t value
        """
        if self.curve_state is None:
            self.load_curve()

        # Load original (reference) model
        original_state = checkpoint.load_state_dict(original_checkpoint)

        # Sample points along the curve
        t_values = np.linspace(0, 1, num_points)

        # Track distances for each layer at each t
        layer_distances = {}

        for t in t_values:
            # Get curve parameters at this t value
            curve_params_t = self.get_params_at_t(t)

            if permutation_invariant:
                # Use permutation-invariant distance (difference of norms)
                distances_t = metrics.permutation_invariant_distance(
                    curve_params_t,
                    original_state
                )
            else:
                # Use standard L2 distance
                distances_t = metrics.layer_wise_l2(
                    curve_params_t,
                    original_state
                )

            # Organize by layer
            for layer_name, distance in distances_t.items():
                if layer_name not in layer_distances:
                    layer_distances[layer_name] = []
                layer_distances[layer_name].append(distance)

        return {
            't_values': t_values.tolist(),
            'layer_distances': {k: v for k, v in layer_distances.items()},
            'num_layers': len(layer_distances),
            'num_points': num_points
        }

    def verify_symmetry_plane(self,
                              endpoint0: str,
                              endpoint1: str,
                              verbose: bool = True) -> Dict[str, Any]:
        """Verify that middle bend point lies on symmetry plane.

        The symmetry plane is defined by the midpoint m = (θ₀ + θ₁)/2
        and normal vector n = θ₁ - θ₀. For perfect symmetry:
        n · (θ_mid - m) = 0

        Args:
            endpoint0: Path to first endpoint checkpoint
            endpoint1: Path to second endpoint checkpoint
            verbose: Whether to print verification details

        Returns:
            Dictionary with symmetry verification statistics
        """
        if not self.bends:
            self.extract_bends()

        # Load endpoints
        state0 = checkpoint.load_state_dict(endpoint0)
        state1 = checkpoint.load_state_dict(endpoint1)

        # Get middle bend (bend 1 of 3)
        if len(self.bends) < 3:
            raise ValueError(f"Expected 3 bends but found {len(self.bends)}")

        mid_bend = self.bends[1]

        # Calculate symmetry plane
        # Midpoint: m = (θ₀ + θ₁) / 2
        # Normal: n = θ₁ - θ₀

        distances_to_plane = {}
        total_dist_squared = 0.0

        for key in mid_bend.keys():
            if key not in state0 or key not in state1:
                continue

            theta0 = state0[key].flatten()
            theta1 = state1[key].flatten()
            theta_mid = mid_bend[key].flatten()

            # Midpoint
            m = (theta0 + theta1) / 2

            # Normal vector
            n = theta1 - theta0

            # Distance to plane: n · (θ_mid - m)
            dist_vector = theta_mid - m
            dist = torch.dot(n, dist_vector).item()

            # Normalized distance
            n_norm = torch.norm(n).item()
            normalized_dist = abs(dist) / n_norm if n_norm > 0 else 0

            distances_to_plane[key] = {
                'distance': dist,
                'normalized_distance': normalized_dist,
                'normal_norm': n_norm
            }

            total_dist_squared += dist ** 2

        total_distance = np.sqrt(total_dist_squared)

        if verbose:
            print(f"\n{'='*70}")
            print("SYMMETRY PLANE VERIFICATION")
            print(f"{'='*70}")
            print(f"\nTotal distance to symmetry plane: {total_distance:.6e}")

            # Show top violating layers
            sorted_layers = sorted(
                distances_to_plane.items(),
                key=lambda x: abs(x[1]['distance']),
                reverse=True
            )

            print(f"\nTop 5 layers furthest from symmetry plane:")
            print("-" * 70)
            for i, (layer_name, stats) in enumerate(sorted_layers[:5]):
                print(f"{i+1}. {layer_name[:50]:<50} {stats['normalized_distance']:.6e}")

            # Determine if symmetric
            tolerance = 1e-4
            is_symmetric = total_distance < tolerance

            if is_symmetric:
                print(f"\n✓ SYMMETRIC: Distance {total_distance:.6e} < {tolerance:.6e}")
            else:
                print(f"\n✗ NOT SYMMETRIC: Distance {total_distance:.6e} ≥ {tolerance:.6e}")

            print("=" * 70)

        return {
            'total_distance': total_distance,
            'layer_distances': distances_to_plane,
            'is_symmetric': total_distance < 1e-4,
            'tolerance': 1e-4
        }

    def compare_with_other_curves(self,
                                  other_checkpoints: List[str],
                                  verbose: bool = True) -> Dict[str, Any]:
        """Compare this curve with others (different seeds/initializations).

        Args:
            other_checkpoints: List of paths to other curve checkpoints
            verbose: Whether to print comparison details

        Returns:
            Dictionary with comparison statistics
        """
        if self.curve_state is None:
            self.load_curve()

        if verbose:
            print(f"\n{'='*70}")
            print(f"COMPARING {len(other_checkpoints) + 1} CURVES")
            print(f"{'='*70}")

        comparisons = []

        for i, other_path in enumerate(other_checkpoints):
            other_state = checkpoint.load_state_dict(other_path)

            # Calculate distances
            distance_stats = metrics.l2_distance(
                self.curve_state,
                other_state,
                compute_per_layer=False
            )

            cosine_sim = metrics.cosine_similarity(
                self.curve_state,
                other_state
            )

            max_diff = metrics.max_absolute_difference(
                self.curve_state,
                other_state
            )

            comparison = {
                'curve_index': i + 1,
                'checkpoint_path': other_path,
                'l2_distance': distance_stats['total_l2'],
                'normalized_l2': distance_stats['normalized_total_l2'],
                'cosine_similarity': cosine_sim,
                'max_absolute_diff': max_diff
            }

            comparisons.append(comparison)

            if verbose:
                print(f"\nCurve {i+1}: {Path(other_path).name}")
                print(f"  L2 distance:        {comparison['l2_distance']:.6f}")
                print(f"  Normalized L2:      {comparison['normalized_l2']:.6f}")
                print(f"  Cosine similarity:  {comparison['cosine_similarity']:.6f}")
                print(f"  Max abs difference: {comparison['max_absolute_diff']:.6e}")

        if verbose:
            print("=" * 70)

        return {
            'num_curves': len(other_checkpoints) + 1,
            'comparisons': comparisons
        }

    def compute_curve_statistics(self) -> Dict[str, Any]:
        """Compute statistics about curve parameters.

        Returns:
            Dictionary with curve statistics
        """
        if not self.bends:
            self.extract_bends()

        if len(self.bends) == 0:
            return {'num_bends': 0}

        # Compute distances between consecutive bends
        bend_distances = []
        for i in range(len(self.bends) - 1):
            dist = metrics.l2_distance(
                self.bends[i],
                self.bends[i+1],
                compute_per_layer=False
            )
            bend_distances.append(dist['total_l2'])

        return {
            'num_bends': len(self.bends),
            'bend_to_bend_distances': bend_distances,
            'total_path_length': sum(bend_distances),
            'mean_bend_distance': np.mean(bend_distances) if bend_distances else 0,
        }
