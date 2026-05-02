"""
Benchmark permutation alignment methods.

Tests if alignment methods can recover known permutations between LMC-connected modes.

Workflow:
1. Load w_0, w_1 (LMC-connected from shared init training)
2. Generate random permutation P*
3. Create w_1' = P*(w_1) - this breaks LMC
4. Run alignment: P_found = method(w_0, w_1')
5. Recover: w_1_recovered = P_found(w_1')
6. Evaluate barriers:
   - barrier(w_0 <-> w_1_recovered) - does alignment restore connectivity to w_0?
   - barrier(w_1 <-> w_1_recovered) - does alignment restore connectivity to original w_1?
7. Compare P_found with inverse(P*)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict

# Add project paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'external/dnn-mode-connectivity'))

import models
import data
import utils

from scripts.lib.transform.random_permutation import VGG16RandomPermutation
from scripts.lib.alignment.permutation_spec import vgg16_permutation_spec
from scripts.lib.alignment.weight_matching import (
    weight_matching, apply_permutation, compare_permutations
)


def load_model(checkpoint_path: str, num_classes: int = 10):
    """Load VGG16 model from checkpoint."""
    model = models.VGG16.base(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    return model


def evaluate_barrier(
    model_a,
    model_b,
    loaders,
    num_points: int = 11,
    device: torch.device = None
):
    """Evaluate linear interpolation barrier between two models.

    Args:
        model_a: First model (t=0)
        model_b: Second model (t=1)
        loaders: Data loaders dict with 'train' and 'test'
        num_points: Number of interpolation points
        device: Device to use

    Returns:
        Dictionary with barrier metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_a = model_a.to(device)
    model_b = model_b.to(device)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    # Create interpolation model
    interp_model = models.VGG16.base(num_classes=10).to(device)

    ts = np.linspace(0, 1, num_points)
    results = {
        't': ts.tolist(),
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
    }

    for t in ts:
        # Linear interpolation of parameters
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1 - t) * state_a[key] + t * state_b[key]
        interp_model.load_state_dict(interp_state)

        # Evaluate
        interp_model.eval()
        train_res = evaluate_model(interp_model, loaders['train'], device)
        test_res = evaluate_model(interp_model, loaders['test'], device)

        results['train_loss'].append(train_res['loss'])
        results['train_acc'].append(train_res['accuracy'])
        results['test_loss'].append(test_res['loss'])
        results['test_acc'].append(test_res['accuracy'])

    # Compute barrier metrics
    endpoint_test_loss = (results['test_loss'][0] + results['test_loss'][-1]) / 2
    max_test_loss = max(results['test_loss'])
    barrier = max_test_loss - endpoint_test_loss

    results['barrier'] = barrier
    results['max_test_loss'] = max_test_loss
    results['endpoint_avg_test_loss'] = endpoint_test_loss

    return results


def evaluate_model(model, loader, device):
    """Evaluate model on data loader."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets, reduction='sum')
            total_loss += loss.item()
            pred = outputs.argmax(dim=1)
            correct += pred.eq(targets).sum().item()
            total += targets.size(0)

    return {
        'loss': total_loss / total,
        'accuracy': 100.0 * correct / total
    }


def state_dict_to_flat_params(state_dict, ps):
    """Convert state_dict to flat params dict matching permutation spec keys."""
    params = {}
    for key in ps.axes_to_perm:
        if key in state_dict:
            params[key] = state_dict[key]
    return params


def compute_l2_distance(state_a, state_b):
    """Compute L2 distance between two state dicts.

    Returns:
        Dictionary with l2_distance, num_params, and rms_difference
    """
    total_diff_sq = 0.0
    total_params = 0
    for key in state_a:
        diff = state_a[key].float() - state_b[key].float()
        total_diff_sq += (diff ** 2).sum().item()
        total_params += diff.numel()

    l2_dist = total_diff_sq ** 0.5
    rms_diff = (total_diff_sq / total_params) ** 0.5

    return {
        'l2_distance': l2_dist,
        'num_params': total_params,
        'rms_difference': rms_diff
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark permutation alignment')
    parser.add_argument('--w0', type=str, required=True, help='Path to w_0 checkpoint')
    parser.add_argument('--w1', type=str, required=True, help='Path to w_1 checkpoint')
    parser.add_argument('--perm-seed', type=int, default=42, help='Seed for random permutation')
    parser.add_argument('--wm-seed', type=int, default=0, help='Seed for weight matching permutation-update order')
    parser.add_argument('--method', type=str, default='weight_matching',
                        choices=['weight_matching'], help='Alignment method')
    parser.add_argument('--max-iter', type=int, default=100, help='Max iterations for weight matching')
    parser.add_argument('--num-eval-points', type=int, default=11, help='Number of interpolation points')
    parser.add_argument('--data-path', type=str, default='./data', help='Path to data')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--output', type=str, default=None, help='Output file for results')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    print("\n" + "=" * 70)
    print("Loading models...")
    print("=" * 70)
    model_w0 = load_model(args.w0)
    model_w1 = load_model(args.w1)
    print(f"w_0: {args.w0}")
    print(f"w_1: {args.w1}")

    # Load data
    print("\nLoading data...")
    loaders, num_classes = data.loaders(
        'CIFAR10',
        args.data_path,
        args.batch_size,
        num_workers=4,
        transform_name='VGG',
        use_test=True,
        eval_mode=True
    )

    # Step 1: Evaluate original barrier and distance
    print("\n" + "=" * 70)
    print("Step 1: Evaluating original barrier (w_0 <-> w_1)")
    print("=" * 70)

    # Compute L2 distance between w_0 and w_1
    dist_w0_w1 = compute_l2_distance(model_w0.state_dict(), model_w1.state_dict())
    print(f"L2 distance (w_0 <-> w_1): {dist_w0_w1['l2_distance']:.4f}")
    print(f"RMS difference: {dist_w0_w1['rms_difference']:.6f}")
    print(f"Total parameters: {dist_w0_w1['num_params']:,}")

    original_barrier = evaluate_barrier(model_w0, model_w1, loaders, args.num_eval_points, device)
    print(f"Original barrier: {original_barrier['barrier']:.4f}")
    print(f"Endpoint avg test loss: {original_barrier['endpoint_avg_test_loss']:.4f}")
    print(f"Max test loss: {original_barrier['max_test_loss']:.4f}")

    # Step 2: Generate random permutation
    print("\n" + "=" * 70)
    print(f"Step 2: Generating random permutation (seed={args.perm_seed})")
    print("=" * 70)
    perm_gen = VGG16RandomPermutation()
    P_star = perm_gen.generate(seed=args.perm_seed)
    P_star_inv = perm_gen.invert(P_star)
    ps = vgg16_permutation_spec()

    def convert_perm_to_apply_format(P_found):
        """Convert P_found keys from 'P_Conv_0' to 'conv_0' format."""
        converted = {}
        for key, val in P_found.items():
            if key.startswith('P_Conv_'):
                new_key = f"conv_{key[7:]}"
            elif key.startswith('P_Dense_'):
                new_key = f"fc_{key[8:]}"
            else:
                new_key = key
            converted[new_key] = val
        return converted

    def convert_perm_to_compare_format(perm):
        """Convert perm keys from 'conv_0' to 'P_Conv_0' format."""
        converted = {}
        for key, val in perm.items():
            if key.startswith('conv_'):
                new_key = f"P_Conv_{key[5:]}"
            elif key.startswith('fc_'):
                new_key = f"P_Dense_{key[3:]}"
            else:
                new_key = key
            converted[new_key] = val
        return converted

    # Step 3: Apply permutation to w_1
    print("\n" + "=" * 70)
    print("Step 3: Applying permutation to w_1 -> w_1'")
    print("=" * 70)
    w1_prime_state = perm_gen.apply_to_state_dict(model_w1.state_dict(), P_star)
    model_w1_prime = models.VGG16.base(num_classes=10)
    model_w1_prime.load_state_dict(w1_prime_state)

    # Compute L2 distance between w_0 and w_1' (should be similar to independently trained models)
    dist_w0_w1_prime = compute_l2_distance(model_w0.state_dict(), w1_prime_state)
    print(f"L2 distance (w_0 <-> w_1'): {dist_w0_w1_prime['l2_distance']:.4f}")
    print(f"RMS difference: {dist_w0_w1_prime['rms_difference']:.6f}")

    # Also show distance between w_1 and w_1' (should be large due to permutation)
    dist_w1_w1_prime = compute_l2_distance(model_w1.state_dict(), w1_prime_state)
    print(f"L2 distance (w_1 <-> w_1'): {dist_w1_w1_prime['l2_distance']:.4f}")
    print(f"RMS difference: {dist_w1_w1_prime['rms_difference']:.6f}")

    # Step 4: Evaluate permuted barrier
    print("\n" + "=" * 70)
    print("Step 4: Evaluating permuted barrier (w_0 <-> w_1')")
    print("=" * 70)
    permuted_barrier = evaluate_barrier(model_w0, model_w1_prime, loaders, args.num_eval_points, device)
    print(f"Permuted barrier: {permuted_barrier['barrier']:.4f}")
    print(f"Endpoint avg test loss: {permuted_barrier['endpoint_avg_test_loss']:.4f}")
    print(f"Max test loss: {permuted_barrier['max_test_loss']:.4f}")

    # Step 5: Run alignment method
    print("\n" + "=" * 70)
    print(f"Step 5: Running alignment ({args.method}): align w_1' to w_0")
    print("=" * 70)
    print(f"Weight matching seed: {args.wm_seed}")

    params_w0 = state_dict_to_flat_params(model_w0.state_dict(), ps)
    params_w1_prime = state_dict_to_flat_params(w1_prime_state, ps)

    P_found = weight_matching(
        ps,
        params_w0,
        params_w1_prime,
        max_iter=args.max_iter,
        seed=args.wm_seed,
        silent=False
    )

    # Step 6: Apply found permutation
    print("\n" + "=" * 70)
    print("Step 6: Applying found permutation to w_1' -> w_1_recovered")
    print("=" * 70)

    P_found_converted = convert_perm_to_apply_format(P_found)
    w1_recovered_state = perm_gen.apply_to_state_dict(w1_prime_state, P_found_converted)
    model_w1_recovered = models.VGG16.base(num_classes=10)
    model_w1_recovered.load_state_dict(w1_recovered_state)

    # Compute distances for recovered model
    dist_w0_w1_rec = compute_l2_distance(model_w0.state_dict(), w1_recovered_state)
    print(f"L2 distance (w_0 <-> w_1_rec): {dist_w0_w1_rec['l2_distance']:.4f}")

    dist_w1_w1_rec = compute_l2_distance(model_w1.state_dict(), w1_recovered_state)
    print(f"L2 distance (w_1 <-> w_1_rec): {dist_w1_w1_rec['l2_distance']:.4f}")

    # Sanity check: verify w_1_rec == w_1 if P_found == P*^{-1}
    # This confirms: P_found(P*(w_1)) == w_1, meaning P_found == P*^{-1}
    w1_state = model_w1.state_dict()
    max_diff = 0.0
    for key in w1_state:
        diff = torch.abs(w1_state[key] - w1_recovered_state[key]).max().item()
        max_diff = max(max_diff, diff)

    print(f"Max element-wise diff (w_1 vs w_1_rec): {max_diff:.2e}")
    if max_diff < 1e-5:
        print("VERIFIED: w_1_recovered == w_1 (P_found recovers exact inverse permutation)")
    else:
        print(f"WARNING: w_1_recovered differs from w_1 (max diff: {max_diff:.6f})")

    # Step 7A: Evaluate recovered barrier to w_0
    print("\n" + "=" * 70)
    print("Step 7A: Evaluating recovered barrier (w_0 <-> w_1_recovered)")
    print("=" * 70)
    recovered_barrier_to_w0 = evaluate_barrier(model_w0, model_w1_recovered, loaders, args.num_eval_points, device)
    print(f"Recovered barrier to w_0: {recovered_barrier_to_w0['barrier']:.4f}")
    print(f"Endpoint avg test loss: {recovered_barrier_to_w0['endpoint_avg_test_loss']:.4f}")
    print(f"Max test loss: {recovered_barrier_to_w0['max_test_loss']:.4f}")

    # Step 7B: Evaluate recovered barrier to original w_1
    print("\n" + "=" * 70)
    print("Step 7B: Evaluating recovered barrier (w_1 <-> w_1_recovered)")
    print("=" * 70)
    recovered_barrier_to_w1 = evaluate_barrier(model_w1, model_w1_recovered, loaders, args.num_eval_points, device)
    print(f"Recovered barrier to w_1: {recovered_barrier_to_w1['barrier']:.4f}")
    print(f"Endpoint avg test loss: {recovered_barrier_to_w1['endpoint_avg_test_loss']:.4f}")
    print(f"Max test loss: {recovered_barrier_to_w1['max_test_loss']:.4f}")

    # Step 8: Compare permutations
    print("\n" + "=" * 70)
    print("Step 8: Comparing P_found with P*^{-1}")
    print("=" * 70)

    P_star_inv_converted = convert_perm_to_compare_format(P_star_inv)
    perm_comparison = compare_permutations(P_found, P_star_inv_converted)
    print(f"Overall permutation accuracy: {perm_comparison['overall_accuracy']:.2%}")
    print("\nPer-layer accuracy:")
    for layer, stats in perm_comparison['per_layer'].items():
        print(f"  {layer}: {stats['accuracy']:.2%} ({stats['matched']}/{stats['total']})")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nL2 DISTANCES:")
    print(f"  w_0 <-> w_1 (original LMC pair):           {dist_w0_w1['l2_distance']:.4f}")
    print(f"  w_0 <-> w_1' (after permutation):          {dist_w0_w1_prime['l2_distance']:.4f}")
    print(f"  w_1 <-> w_1' (permutation effect):         {dist_w1_w1_prime['l2_distance']:.4f}")
    print(f"  w_0 <-> w_1_rec (after alignment):         {dist_w0_w1_rec['l2_distance']:.4f}")
    print(f"  w_1 <-> w_1_rec (recovery accuracy):       {dist_w1_w1_rec['l2_distance']:.4f}")

    print("\nBARRIERS:")
    print(f"  Original (w_0 <-> w_1):                    {original_barrier['barrier']:.4f}")
    print(f"  Permuted (w_0 <-> w_1'):                   {permuted_barrier['barrier']:.4f}")
    print(f"  Recovered to w_0 (w_0 <-> w_1_rec):        {recovered_barrier_to_w0['barrier']:.4f}")
    print(f"  Recovered to w_1 (w_1 <-> w_1_rec):        {recovered_barrier_to_w1['barrier']:.4f}")

    print(f"\nPermutation recovery accuracy:               {perm_comparison['overall_accuracy']:.2%}")
    print()

    # Evaluate success
    if recovered_barrier_to_w0['barrier'] < permuted_barrier['barrier'] * 0.5:
        print("Connectivity to w_0: SUCCESS - Alignment significantly reduced barrier!")
    elif recovered_barrier_to_w0['barrier'] < permuted_barrier['barrier'] * 0.9:
        print("Connectivity to w_0: PARTIAL - Alignment partially reduced barrier")
    else:
        print("Connectivity to w_0: FAILURE - Alignment did not reduce barrier")

    if recovered_barrier_to_w1['barrier'] < 0.1:
        print("Connectivity to w_1: SUCCESS - w_1_recovered is close to original w_1!")
    elif recovered_barrier_to_w1['barrier'] < 0.5:
        print("Connectivity to w_1: PARTIAL - w_1_recovered partially matches w_1")
    else:
        print("Connectivity to w_1: FAILURE - w_1_recovered is far from original w_1")

    # Save results
    if args.output:
        import json
        results = {
            'config': vars(args),
            'distances': {
                'w0_w1': dist_w0_w1,
                'w0_w1_prime': dist_w0_w1_prime,
                'w1_w1_prime': dist_w1_w1_prime,
                'w0_w1_recovered': dist_w0_w1_rec,
                'w1_w1_recovered': dist_w1_w1_rec,
            },
            'original_barrier': original_barrier,
            'permuted_barrier': permuted_barrier,
            'recovered_barrier_to_w0': recovered_barrier_to_w0,
            'recovered_barrier_to_w1': recovered_barrier_to_w1,
            'permutation_comparison': {
                'overall_accuracy': perm_comparison['overall_accuracy'],
                'per_layer': {k: v['accuracy'] for k, v in perm_comparison['per_layer'].items()}
            }
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
