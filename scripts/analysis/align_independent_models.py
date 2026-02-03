"""
Align independently trained models using weight matching.

Tests if weight matching can find a permutation that reduces the barrier
between two independently trained models (no shared initialization).

Workflow:
1. Load seed0, seed1 (independently trained)
2. Evaluate original barrier (expected high)
3. Run alignment: P = weight_matching(seed0, seed1)
4. Apply permutation: seed1_aligned = P(seed1)
5. Evaluate aligned barrier
6. Compare distances and barriers
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

from scripts.lib.alignment.permutation_spec import vgg16_permutation_spec
from scripts.lib.alignment.weight_matching import weight_matching
from scripts.lib.transform.random_permutation import VGG16RandomPermutation


def load_model(checkpoint_path: str, num_classes: int = 10):
    """Load VGG16 model from checkpoint."""
    model = models.VGG16.base(num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state'])
    return model


def compute_l2_distance(state_a, state_b):
    """Compute L2 distance between two state dicts."""
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


def evaluate_barrier(
    model_a,
    model_b,
    loaders,
    num_points: int = 61,
    device: torch.device = None
):
    """Evaluate linear interpolation barrier between two models."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_a = model_a.to(device)
    model_b = model_b.to(device)

    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

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
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1 - t) * state_a[key] + t * state_b[key]
        interp_model.load_state_dict(interp_state)

        interp_model.eval()
        train_res = evaluate_model(interp_model, loaders['train'], device)
        test_res = evaluate_model(interp_model, loaders['test'], device)

        results['train_loss'].append(train_res['loss'])
        results['train_acc'].append(train_res['accuracy'])
        results['test_loss'].append(test_res['loss'])
        results['test_acc'].append(test_res['accuracy'])

    endpoint_test_loss = (results['test_loss'][0] + results['test_loss'][-1]) / 2
    max_test_loss = max(results['test_loss'])
    min_test_acc = min(results['test_acc'])
    barrier = max_test_loss - endpoint_test_loss

    results['barrier'] = barrier
    results['max_test_loss'] = max_test_loss
    results['min_test_acc'] = min_test_acc
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


def main():
    parser = argparse.ArgumentParser(description='Align independently trained models')
    parser.add_argument('--model-a', type=str, required=True, help='Path to first model (reference)')
    parser.add_argument('--model-b', type=str, required=True, help='Path to second model (to be aligned)')
    parser.add_argument('--method', type=str, default='weight_matching',
                        choices=['weight_matching'], help='Alignment method')
    parser.add_argument('--max-iter', type=int, default=100, help='Max iterations for weight matching')
    parser.add_argument('--num-eval-points', type=int, default=21, help='Number of interpolation points')
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
    model_a = load_model(args.model_a)
    model_b = load_model(args.model_b)
    print(f"Model A (reference): {args.model_a}")
    print(f"Model B (to align):  {args.model_b}")

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

    # =========================================================================
    # Step 1: Evaluate original (before alignment)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Evaluating BEFORE alignment (A <-> B)")
    print("=" * 70)

    dist_original = compute_l2_distance(model_a.state_dict(), model_b.state_dict())
    print(f"L2 distance (A <-> B): {dist_original['l2_distance']:.4f}")
    print(f"RMS difference: {dist_original['rms_difference']:.6f}")
    print(f"Total parameters: {dist_original['num_params']:,}")

    barrier_original = evaluate_barrier(model_a, model_b, loaders, args.num_eval_points, device)
    print(f"\nOriginal barrier: {barrier_original['barrier']:.4f}")
    print(f"Endpoint avg test loss: {barrier_original['endpoint_avg_test_loss']:.4f}")
    print(f"Max test loss: {barrier_original['max_test_loss']:.4f}")
    print(f"Min test accuracy along path: {barrier_original['min_test_acc']:.2f}%")

    # =========================================================================
    # Step 2: Run alignment
    # =========================================================================
    print("\n" + "=" * 70)
    print(f"Step 2: Running alignment ({args.method})")
    print("=" * 70)

    ps = vgg16_permutation_spec()
    params_a = state_dict_to_flat_params(model_a.state_dict(), ps)
    params_b = state_dict_to_flat_params(model_b.state_dict(), ps)

    P_found = weight_matching(
        ps,
        params_a,
        params_b,
        max_iter=args.max_iter,
        silent=False
    )

    # =========================================================================
    # Step 3: Apply permutation to model B
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 3: Applying permutation to B -> B_aligned")
    print("=" * 70)

    perm_gen = VGG16RandomPermutation()
    P_found_converted = convert_perm_to_apply_format(P_found)
    b_aligned_state = perm_gen.apply_to_state_dict(model_b.state_dict(), P_found_converted)
    model_b_aligned = models.VGG16.base(num_classes=10)
    model_b_aligned.load_state_dict(b_aligned_state)

    # Verify B_aligned has same accuracy as B (permutation preserves function)
    print("Verifying B_aligned accuracy...")
    b_acc = evaluate_model(model_b.to(device), loaders['test'], device)
    b_aligned_acc = evaluate_model(model_b_aligned.to(device), loaders['test'], device)
    print(f"  Model B accuracy: {b_acc['accuracy']:.2f}%")
    print(f"  Model B_aligned accuracy: {b_aligned_acc['accuracy']:.2f}%")
    if abs(b_acc['accuracy'] - b_aligned_acc['accuracy']) < 0.01:
        print("  VERIFIED: Permutation preserves model function")
    else:
        print("  WARNING: Accuracy changed after permutation!")

    # =========================================================================
    # Step 4: Evaluate after alignment
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Evaluating AFTER alignment (A <-> B_aligned)")
    print("=" * 70)

    dist_aligned = compute_l2_distance(model_a.state_dict(), b_aligned_state)
    print(f"L2 distance (A <-> B_aligned): {dist_aligned['l2_distance']:.4f}")
    print(f"RMS difference: {dist_aligned['rms_difference']:.6f}")

    barrier_aligned = evaluate_barrier(model_a, model_b_aligned, loaders, args.num_eval_points, device)
    print(f"\nAligned barrier: {barrier_aligned['barrier']:.4f}")
    print(f"Endpoint avg test loss: {barrier_aligned['endpoint_avg_test_loss']:.4f}")
    print(f"Max test loss: {barrier_aligned['max_test_loss']:.4f}")
    print(f"Min test accuracy along path: {barrier_aligned['min_test_acc']:.2f}%")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nL2 DISTANCES:")
    print(f"  Before alignment (A <-> B):         {dist_original['l2_distance']:.4f}")
    print(f"  After alignment (A <-> B_aligned):  {dist_aligned['l2_distance']:.4f}")
    print(f"  Change: {dist_aligned['l2_distance'] - dist_original['l2_distance']:+.4f} ({(dist_aligned['l2_distance']/dist_original['l2_distance'] - 1)*100:+.1f}%)")

    print("\nBARRIERS:")
    print(f"  Before alignment: {barrier_original['barrier']:.4f} (min acc: {barrier_original['min_test_acc']:.2f}%)")
    print(f"  After alignment:  {barrier_aligned['barrier']:.4f} (min acc: {barrier_aligned['min_test_acc']:.2f}%)")

    barrier_reduction = (barrier_original['barrier'] - barrier_aligned['barrier']) / barrier_original['barrier'] * 100
    print(f"  Barrier reduction: {barrier_reduction:.1f}%")
    print()

    # Evaluate success
    if barrier_aligned['barrier'] < 0.1:
        print("RESULT: SUCCESS - Models are now LMC-connected!")
    elif barrier_aligned['barrier'] < barrier_original['barrier'] * 0.5:
        print("RESULT: PARTIAL SUCCESS - Alignment significantly reduced barrier")
    elif barrier_aligned['barrier'] < barrier_original['barrier'] * 0.9:
        print("RESULT: MARGINAL - Alignment partially reduced barrier")
    else:
        print("RESULT: FAILURE - Alignment did not reduce barrier significantly")

    # Save results
    if args.output:
        import json
        results = {
            'config': vars(args),
            'before_alignment': {
                'distance': dist_original,
                'barrier': barrier_original,
            },
            'after_alignment': {
                'distance': dist_aligned,
                'barrier': barrier_aligned,
            },
            'barrier_reduction_percent': barrier_reduction,
        }
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
