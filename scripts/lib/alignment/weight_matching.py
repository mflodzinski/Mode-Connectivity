"""
Weight matching algorithm for finding permutation alignment between models.

Adapted from git-rebasin (JAX) to PyTorch.
"""

import numpy as np
import torch
from typing import Dict, Optional
from scipy.optimize import linear_sum_assignment

from .permutation_spec import PermutationSpec


def get_permuted_param(
    ps: PermutationSpec,
    perm: Dict[str, np.ndarray],
    key: str,
    params: Dict[str, torch.Tensor],
    except_axis: Optional[int] = None
) -> torch.Tensor:
    """Get parameter with permutations applied.

    Args:
        ps: Permutation specification
        perm: Current permutation dictionary
        key: Parameter key to retrieve
        params: Parameter dictionary
        except_axis: Axis to skip (used when computing cost matrix)

    Returns:
        Permuted parameter tensor
    """
    w = params[key]

    for axis, p in enumerate(ps.axes_to_perm[key]):
        # Skip the axis we're trying to permute
        if axis == except_axis:
            continue

        # None means no permutation for that axis
        if p is not None:
            # torch.index_select equivalent to jnp.take
            perm_indices = torch.tensor(perm[p], dtype=torch.long, device=w.device)
            w = torch.index_select(w, dim=axis, index=perm_indices)

    return w


def apply_permutation(
    ps: PermutationSpec,
    perm: Dict[str, np.ndarray],
    params: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Apply permutation to all parameters.

    Args:
        ps: Permutation specification
        perm: Permutation dictionary
        params: Parameter dictionary

    Returns:
        New parameter dictionary with permutation applied
    """
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(
    ps: PermutationSpec,
    params_a: Dict[str, torch.Tensor],
    params_b: Dict[str, torch.Tensor],
    max_iter: int = 100,
    init_perm: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 0,
    silent: bool = False
) -> Dict[str, np.ndarray]:
    """Find permutation of params_b to match params_a.

    Uses the iterative weight matching algorithm from git-rebasin.
    Each iteration:
    1. For each permutation (in random order):
       - Compute cost matrix based on weight correlations
       - Solve assignment problem using Hungarian algorithm
       - Update permutation if it improves objective

    Args:
        ps: Permutation specification
        params_a: Reference parameters (target)
        params_b: Parameters to align (will be permuted)
        max_iter: Maximum iterations
        init_perm: Initial permutation (default: identity)
        seed: Random seed for iteration order
        silent: Suppress progress output

    Returns:
        Dictionary mapping permutation names to index arrays
    """
    # Get permutation sizes from parameter shapes
    perm_sizes = {}
    for p, axes in ps.perm_to_axes.items():
        wk, axis = axes[0]  # Use first parameter to get size
        perm_sizes[p] = params_a[wk].shape[axis]

    # Initialize permutation (identity if not provided)
    if init_perm is None:
        perm = {p: np.arange(n) for p, n in perm_sizes.items()}
    else:
        perm = {p: np.array(v) for p, v in init_perm.items()}

    perm_names = list(perm.keys())
    rng = np.random.RandomState(seed)

    for iteration in range(max_iter):
        progress = False

        # Random order of permutations each iteration
        perm_order = rng.permutation(len(perm_names))

        for p_ix in perm_order:
            p = perm_names[p_ix]
            n = perm_sizes[p]

            # Compute cost matrix A
            # A[i,j] = sum over all affected weights of correlation between
            # unit i in model_a and unit j in model_b (after current permutation)
            A = np.zeros((n, n))

            for wk, axis in ps.perm_to_axes[p]:
                w_a = params_a[wk]
                # Get w_b with all permutations except the one we're optimizing
                w_b = get_permuted_param(ps, perm, wk, params_b, except_axis=axis)

                # Move the axis we're permuting to position 0, then flatten rest
                # This gives us (n, -1) shaped arrays
                w_a = torch.movedim(w_a, axis, 0).reshape(n, -1)
                w_b = torch.movedim(w_b, axis, 0).reshape(n, -1)

                # Correlation matrix: A += w_a @ w_b.T
                A += (w_a @ w_b.T).cpu().numpy()

            # Solve assignment problem
            ri, ci = linear_sum_assignment(A, maximize=True)
            assert np.all(ri == np.arange(len(ri)))

            # Check if new assignment improves objective
            old_perm = perm[p]
            old_score = np.sum(A[np.arange(n), old_perm])
            new_score = np.sum(A[np.arange(n), ci])

            if not silent:
                print(f"{iteration}/{p}: {new_score - old_score:.6f}")

            if new_score > old_score + 1e-12:
                progress = True
                perm[p] = ci

        if not progress:
            if not silent:
                print(f"Converged after {iteration + 1} iterations")
            break

    return perm


def compare_permutations(
    perm1: Dict[str, np.ndarray],
    perm2: Dict[str, np.ndarray]
) -> Dict[str, float]:
    """Compare two permutations and compute match statistics.

    Args:
        perm1: First permutation
        perm2: Second permutation

    Returns:
        Dictionary with comparison metrics
    """
    results = {
        'per_layer': {},
        'total_matched': 0,
        'total_elements': 0,
    }

    for key in perm1:
        if key not in perm2:
            continue

        p1, p2 = perm1[key], perm2[key]
        n = len(p1)
        matched = np.sum(p1 == p2)

        results['per_layer'][key] = {
            'matched': int(matched),
            'total': n,
            'accuracy': matched / n
        }
        results['total_matched'] += matched
        results['total_elements'] += n

    if results['total_elements'] > 0:
        results['overall_accuracy'] = results['total_matched'] / results['total_elements']
    else:
        results['overall_accuracy'] = 0.0

    return results
