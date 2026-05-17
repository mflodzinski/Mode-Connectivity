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

import argparse
import numpy as np
import torch

from mode_connectivity.transform.random_permutation import VGG16RandomPermutation
from mode_connectivity.alignment.permutation_spec import (
    vgg16_features_permutation_spec,
    vgg16_permutation_spec,
)
from mode_connectivity.alignment.weight_matching import (
    weight_matching, apply_permutation, compare_permutations
)
from mode_connectivity.analysis.alignment import (
    convert_perm_to_compare_format,
    max_abs_state_diff,
    state_dict_to_perm_params,
)
from mode_connectivity.core import data as core_data
from mode_connectivity.core.checkpoint import (
    build_model_from_state_dict,
    load_checkpoint_state,
)
from mode_connectivity.evaluation.interpolation import evaluate_linear_interpolation
from mode_connectivity.evaluation.metrics import state_distance_summary


def load_model(checkpoint_path: str, num_classes: int = 10):
    """Load VGG16 model from checkpoint and detect its format."""
    state_dict, checkpoint_format = load_checkpoint_state(checkpoint_path)
    model = build_model_from_state_dict(
        state_dict,
        checkpoint_family=checkpoint_format,
        num_classes=num_classes,
    )
    return model, state_dict, checkpoint_format


def evaluate_barrier(
    model_a,
    model_b,
    loaders,
    num_points: int = 11,
    device: torch.device = None,
    checkpoint_format: str = 'dnn_mode_connectivity',
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

    interp_model = build_model_from_state_dict(
        state_a,
        checkpoint_family=checkpoint_format,
        num_classes=10,
    ).to(device)

    results = evaluate_linear_interpolation(
        state_a=state_a,
        state_b=state_b,
        model=interp_model,
        train_loader=loaders['train'],
        test_loader=loaders['test'],
        device=device,
        ts=np.linspace(0, 1, num_points),
    )

    # Compute barrier metrics
    endpoint_test_loss = (results['test_loss'][0] + results['test_loss'][-1]) / 2
    max_test_loss = max(results['test_loss'])
    barrier = max_test_loss - endpoint_test_loss

    results['barrier'] = barrier
    results['max_test_loss'] = max_test_loss
    results['endpoint_avg_test_loss'] = endpoint_test_loss

    return results


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
    model_w0, state_w0, format_w0 = load_model(args.w0)
    model_w1, state_w1, format_w1 = load_model(args.w1)
    if format_w0 != format_w1:
        raise ValueError(f"Checkpoint format mismatch: {format_w0} vs {format_w1}")
    checkpoint_format = format_w0
    print(f"w_0: {args.w0}")
    print(f"w_1: {args.w1}")
    print(f"Checkpoint format: {checkpoint_format}")

    # Load data
    print("\nLoading data...")
    loaders, num_classes = core_data.get_loaders(
        'CIFAR10',
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=4,
        transform_name='VGG',
        use_test=True,
        shuffle_train=False,
    )

    # Step 1: Evaluate original barrier and distance
    print("\n" + "=" * 70)
    print("Step 1: Evaluating original barrier (w_0 <-> w_1)")
    print("=" * 70)

    # Compute L2 distance between w_0 and w_1
    dist_w0_w1 = state_distance_summary(state_w0, state_w1)
    print(f"L2 distance (w_0 <-> w_1): {dist_w0_w1['l2_distance']:.4f}")
    print(f"RMS difference: {dist_w0_w1['rms_difference']:.6f}")
    print(f"Total parameters: {dist_w0_w1['num_params']:,}")

    original_barrier = evaluate_barrier(
        model_w0, model_w1, loaders, args.num_eval_points, device, checkpoint_format
    )
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
    if checkpoint_format == 'dnn_mode_connectivity':
        ps = vgg16_permutation_spec()
    elif checkpoint_format == 'pytorch_vgg_cifar10':
        ps = vgg16_features_permutation_spec()
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_format}")

    # Step 3: Apply permutation to w_1
    print("\n" + "=" * 70)
    print("Step 3: Applying permutation to w_1 -> w_1'")
    print("=" * 70)
    P_star_spec = convert_perm_to_compare_format(P_star)
    w1_prime_state = apply_permutation(ps, P_star_spec, state_w1)
    model_w1_prime = build_model_from_state_dict(
        w1_prime_state,
        checkpoint_family=checkpoint_format,
        num_classes=10,
    )

    # Compute L2 distance between w_0 and w_1' (should be similar to independently trained models)
    dist_w0_w1_prime = state_distance_summary(state_w0, w1_prime_state)
    print(f"L2 distance (w_0 <-> w_1'): {dist_w0_w1_prime['l2_distance']:.4f}")
    print(f"RMS difference: {dist_w0_w1_prime['rms_difference']:.6f}")

    # Also show distance between w_1 and w_1' (should be large due to permutation)
    dist_w1_w1_prime = state_distance_summary(state_w1, w1_prime_state)
    print(f"L2 distance (w_1 <-> w_1'): {dist_w1_w1_prime['l2_distance']:.4f}")
    print(f"RMS difference: {dist_w1_w1_prime['rms_difference']:.6f}")

    # Step 4: Evaluate permuted barrier
    print("\n" + "=" * 70)
    print("Step 4: Evaluating permuted barrier (w_0 <-> w_1')")
    print("=" * 70)
    permuted_barrier = evaluate_barrier(
        model_w0, model_w1_prime, loaders, args.num_eval_points, device, checkpoint_format
    )
    print(f"Permuted barrier: {permuted_barrier['barrier']:.4f}")
    print(f"Endpoint avg test loss: {permuted_barrier['endpoint_avg_test_loss']:.4f}")
    print(f"Max test loss: {permuted_barrier['max_test_loss']:.4f}")

    # Step 5: Run alignment method
    print("\n" + "=" * 70)
    print(f"Step 5: Running alignment ({args.method}): align w_1' to w_0")
    print("=" * 70)
    print(f"Weight matching seed: {args.wm_seed}")

    params_w0 = state_dict_to_perm_params(state_w0, ps)
    params_w1_prime = state_dict_to_perm_params(w1_prime_state, ps)

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

    w1_recovered_state = apply_permutation(ps, P_found, w1_prime_state)
    model_w1_recovered = build_model_from_state_dict(
        w1_recovered_state,
        checkpoint_family=checkpoint_format,
        num_classes=10,
    )

    # Compute distances for recovered model
    dist_w0_w1_rec = state_distance_summary(state_w0, w1_recovered_state)
    print(f"L2 distance (w_0 <-> w_1_rec): {dist_w0_w1_rec['l2_distance']:.4f}")

    dist_w1_w1_rec = state_distance_summary(state_w1, w1_recovered_state)
    print(f"L2 distance (w_1 <-> w_1_rec): {dist_w1_w1_rec['l2_distance']:.4f}")

    # Sanity check: verify w_1_rec == w_1 if P_found == P*^{-1}
    # This confirms: P_found(P*(w_1)) == w_1, meaning P_found == P*^{-1}
    max_diff = max_abs_state_diff(state_w1, w1_recovered_state)

    print(f"Max element-wise diff (w_1 vs w_1_rec): {max_diff:.2e}")
    if max_diff < 1e-5:
        print("VERIFIED: w_1_recovered == w_1 (P_found recovers exact inverse permutation)")
    else:
        print(f"WARNING: w_1_recovered differs from w_1 (max diff: {max_diff:.6f})")

    # Step 7A: Evaluate recovered barrier to w_0
    print("\n" + "=" * 70)
    print("Step 7A: Evaluating recovered barrier (w_0 <-> w_1_recovered)")
    print("=" * 70)
    recovered_barrier_to_w0 = evaluate_barrier(
        model_w0, model_w1_recovered, loaders, args.num_eval_points, device, checkpoint_format
    )
    print(f"Recovered barrier to w_0: {recovered_barrier_to_w0['barrier']:.4f}")
    print(f"Endpoint avg test loss: {recovered_barrier_to_w0['endpoint_avg_test_loss']:.4f}")
    print(f"Max test loss: {recovered_barrier_to_w0['max_test_loss']:.4f}")

    # Step 7B: Evaluate recovered barrier to original w_1
    print("\n" + "=" * 70)
    print("Step 7B: Evaluating recovered barrier (w_1 <-> w_1_recovered)")
    print("=" * 70)
    recovered_barrier_to_w1 = evaluate_barrier(
        model_w1, model_w1_recovered, loaders, args.num_eval_points, device, checkpoint_format
    )
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
