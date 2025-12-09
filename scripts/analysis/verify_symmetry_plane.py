"""
Verify that middle points lie (or don't lie) on the symmetry plane.

The symmetry plane is defined by: n · (θ - m) = 0
where:
  - n = w₁ - w₀ (normal vector)
  - m = (w₀ + w₁) / 2 (midpoint)
  - θ = middle bend parameters

For a point to be on the plane, the dot product n · (θ - m) should be zero.
"""

import torch
import numpy as np
import argparse
from pathlib import Path


def load_checkpoint(path):
    """Load checkpoint and extract model state."""
    ckpt = torch.load(path, map_location='cpu')
    # Handle both formats: direct state dict or nested under 'model_state'
    if 'model_state' in ckpt:
        return ckpt['model_state']
    return ckpt


def extract_bends_from_curve(curve_state):
    """
    Extract the 3 bends from a curve checkpoint.
    
    Curve models store parameters with suffixes: weight_0, weight_1, weight_2, etc.
    """
    bends = [{}, {}, {}]  # Three bends as dictionaries
    
    for key, value in curve_state.items():
        if key.startswith('net.'):
            # Remove 'net.' prefix
            param_name = key[4:]
            
            # Check if it ends with _0, _1, or _2
            if param_name.endswith('_0'):
                base_name = param_name[:-2]
                bends[0][base_name] = value
            elif param_name.endswith('_1'):
                base_name = param_name[:-2]
                bends[1][base_name] = value
            elif param_name.endswith('_2'):
                base_name = param_name[:-2]
                bends[2][base_name] = value
    
    # Convert to parameter vectors
    bend_vectors = []
    for bend_dict in bends:
        bend_vector = torch.cat([p.flatten() for p in bend_dict.values() if isinstance(p, torch.Tensor)])
        bend_vectors.append(bend_vector)
    
    return bend_vectors


def compute_symmetry_plane_distance(w0, w1, theta):
    """
    Compute distance from point theta to the symmetry plane defined by w0 and w1.
    
    The plane equation is: n · (θ - m) = 0
    where n = w1 - w0 and m = (w0 + w1) / 2
    
    Returns:
        - distance: scalar projection onto normal (should be ~0 if on plane)
        - normalized_distance: distance normalized by ||n||
        - midpoint_distance: L2 distance from theta to midpoint m
    """
    # Compute normal vector and midpoint
    n = w1 - w0  # Normal vector
    m = (w0 + w1) / 2  # Midpoint
    
    # Compute displacement from midpoint
    displacement = theta - m
    
    # Compute dot product: n · (θ - m)
    dot_product = torch.sum(n * displacement).item()
    
    # Compute norms
    n_norm = torch.norm(n).item()
    displacement_norm = torch.norm(displacement).item()
    
    # Normalized distance (projection onto unit normal)
    normalized_distance = dot_product / n_norm if n_norm > 0 else float('inf')
    
    return {
        'dot_product': dot_product,
        'normalized_distance': normalized_distance,
        'normal_norm': n_norm,
        'displacement_norm': displacement_norm,
        'midpoint_distance': displacement_norm,
    }


def main():
    parser = argparse.ArgumentParser(description='Verify symmetry plane constraint')
    parser.add_argument('--endpoint0', type=str, required=True, help='Path to first endpoint')
    parser.add_argument('--endpoint1', type=str, required=True, help='Path to second endpoint')
    parser.add_argument('--curve', type=str, required=True, help='Path to curve checkpoint')
    parser.add_argument('--name', type=str, required=True, help='Name of the experiment (polygon/symmetry_plane)')
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"SYMMETRY PLANE VERIFICATION: {args.name}")
    print("="*80)
    
    # Load endpoints
    print("\nLoading endpoints...")
    w0_state = load_checkpoint(args.endpoint0)
    w1_state = load_checkpoint(args.endpoint1)
    
    # Convert to parameter vectors
    w0 = torch.cat([p.flatten() for p in w0_state.values() if isinstance(p, torch.Tensor)])
    w1 = torch.cat([p.flatten() for p in w1_state.values() if isinstance(p, torch.Tensor)])
    
    print(f"Endpoint 0: {args.endpoint0}")
    print(f"Endpoint 1: {args.endpoint1}")
    print(f"Parameter dimension: {w0.shape[0]:,}")
    
    # Load curve and extract bends
    print(f"\nLoading curve: {args.curve}")
    curve_state = load_checkpoint(args.curve)
    
    # Extract the three bends
    bends = extract_bends_from_curve(curve_state)
    bend0, bend1, bend2 = bends
    
    print(f"Bend 0 (start) dimension: {bend0.shape[0]:,}")
    print(f"Bend 1 (middle) dimension: {bend1.shape[0]:,}")
    print(f"Bend 2 (end) dimension: {bend2.shape[0]:,}")
    
    # Verify dimensions match
    if bend0.shape[0] != w0.shape[0]:
        print(f"\n⚠ WARNING: Dimension mismatch!")
        print(f"  Endpoint: {w0.shape[0]:,} parameters")
        print(f"  Curve:    {bend0.shape[0]:,} parameters")
        print(f"  Difference: {abs(w0.shape[0] - bend0.shape[0]):,} parameters")
        print("\nThis might be due to BatchNorm running statistics not being included.")
    
    # Verify endpoints match (approximately, since BN stats might differ)
    start_diff = torch.norm(bend0 - w0[:bend0.shape[0]]).item()
    end_diff = torch.norm(bend2 - w1[:bend2.shape[0]]).item()
    print(f"\nEndpoint verification (using first {bend0.shape[0]:,} params):")
    print(f"  ||bend0 - w0||: {start_diff:.6e} (should be ~0)")
    print(f"  ||bend2 - w1||: {end_diff:.6e} (should be ~0)")
    
    # Compute symmetry plane distance using curve parameter space
    print("\n" + "-"*80)
    print("SYMMETRY PLANE DISTANCE")
    print("-"*80)
    
    # Use only the parameters that exist in the curve
    w0_curve = w0[:bend0.shape[0]]
    w1_curve = w1[:bend1.shape[0]]
    
    result = compute_symmetry_plane_distance(w0_curve, w1_curve, bend1)
    
    print(f"\nPlane equation: n · (θ - m) = 0")
    print(f"  where n = w1 - w0 (normal)")
    print(f"        m = (w0 + w1) / 2 (midpoint)")
    print(f"        θ = middle bend parameters")
    
    print(f"\nResults:")
    print(f"  Dot product n · (θ - m):     {result['dot_product']:+.6e}")
    print(f"  Normalized distance:          {result['normalized_distance']:+.6e}")
    print(f"  Normal norm ||n||:            {result['normal_norm']:.6f}")
    print(f"  Displacement norm ||θ - m||:  {result['displacement_norm']:.6f}")
    
    # Determine if on plane (using threshold)
    threshold = 1e-3  # Relative threshold
    is_on_plane = abs(result['normalized_distance']) < threshold
    
    print("\n" + "-"*80)
    print("VERDICT")
    print("-"*80)
    if is_on_plane:
        print(f"✓ Middle point IS on the symmetry plane")
        print(f"  (normalized distance < {threshold})")
    else:
        print(f"✗ Middle point IS NOT on the symmetry plane")
        print(f"  (normalized distance >= {threshold})")
    
    # Expected results
    print("\n" + "-"*80)
    print("EXPECTED RESULTS")
    print("-"*80)
    if 'polygon' in args.name.lower():
        print("Polygon chain: Middle point should NOT be on plane")
        if not is_on_plane:
            print("✓ CORRECT: Polygon chain middle is NOT constrained")
        else:
            print("✗ ERROR: Polygon chain should not be on plane!")
    elif 'symmetry' in args.name.lower() or 'symplane' in args.name.lower():
        print("Symmetry plane: Middle point SHOULD be on plane")
        if is_on_plane:
            print("✓ CORRECT: Symmetry plane projection working")
        else:
            print("✗ ERROR: Symmetry plane middle should be on plane!")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
