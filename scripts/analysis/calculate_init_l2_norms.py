#!/usr/bin/env python3
"""Calculate L2 norms of initial middle control points for initialization experiments."""

import os
import sys
import numpy as np
import torch

# Add paths to external modules
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)
external_path = os.path.join(scripts_root, '..', 'external', 'dnn-mode-connectivity')
sys.path.insert(0, external_path)

import curves as curve_module
import models as model_module

def load_endpoint(checkpoint_path):
    """Load endpoint checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint['model_state']

def create_curve_model(num_classes=10, architecture='VGG16'):
    """Create a curve model."""
    arch = getattr(model_module, architecture)
    curve_class = getattr(curve_module, 'Bezier')

    model = curve_module.CurveNet(
        num_classes,
        curve_class,
        arch.curve,
        num_bends=3,
        fix_start=True,
        fix_end=True,
        architecture_kwargs=arch.kwargs
    )

    return model, arch

def import_endpoints(model, architecture, endpoint0_state, endpoint1_state, num_classes=10):
    """Import endpoints into curve model."""
    # Create base model and load endpoint 0
    base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    base_model.load_state_dict(endpoint0_state)
    model.import_base_parameters(base_model, index=0)

    # Create base model and load endpoint 1
    base_model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    base_model.load_state_dict(endpoint1_state)
    model.import_base_parameters(base_model, index=2)

    return model

def extract_middle_point(model):
    """Extract middle control point from curve model."""
    middle_params = []
    for name, param in model.named_parameters():
        if '_1' in name:  # Middle bend has suffix '_1'
            middle_params.append(param.data.flatten())

    # Concatenate all parameters into single vector
    middle_vector = torch.cat(middle_params)
    return middle_vector

def extract_middle_point_from_checkpoint(checkpoint_path):
    """Extract middle control point directly from a checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint['model_state']

    middle_params = []
    for key in sorted(model_state.keys()):
        if key.endswith('_1'):  # Middle bend has suffix '_1'
            middle_params.append(model_state[key].flatten())

    # Concatenate all parameters into single vector
    middle_vector = torch.cat(middle_params)
    return middle_vector

def recreate_initial_middle_point(endpoint0_path, endpoint1_path, init_method, init_params, seed=1):
    """Recreate initial middle control point using specified initialization method.

    Returns:
        tuple: (middle_point, interpolated_l2_norm)
            - middle_point: Flattened tensor of raw middle point parameters
            - interpolated_l2_norm: L2 norm of interpolated weights at t=0.5
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Load endpoints
    endpoint0_state = load_endpoint(endpoint0_path)
    endpoint1_state = load_endpoint(endpoint1_path)

    # Create curve model
    model, arch = create_curve_model(num_classes=10, architecture='VGG16')

    # Import endpoints
    import_endpoints(model, arch, endpoint0_state, endpoint1_state, num_classes=10)

    # Initialize middle point based on method
    if init_method == 'biased':
        alpha = init_params.get('alpha', 0.5)
        model.init_linear_custom(alpha=alpha)
    elif init_method == 'perturbed':
        alpha = init_params.get('alpha', 0.5)
        noise = init_params.get('noise', 0.01)
        model.init_perturbed_linear(alpha=alpha, noise_scale=noise)
    elif init_method == 'sphere':
        alpha = init_params.get('alpha', 0.5)
        noise = init_params.get('noise', 0.01)
        inside = init_params.get('inside', True)
        model.init_sphere_constrained(alpha=alpha, noise_scale=noise, inside=inside)
    else:
        raise ValueError(f"Unknown initialization method: {init_method}")

    # Extract middle point (raw parameters)
    middle_point = extract_middle_point(model)

    # Calculate interpolated L2 norm at t=0.5
    t = torch.FloatTensor([0.5])
    weights = model.weights(t)
    interpolated_l2_norm = np.sqrt(np.sum(np.square(weights)))

    return middle_point, interpolated_l2_norm

def main():
    # Paths to endpoints
    endpoint0_path = 'results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt'
    endpoint1_path = 'results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt'

    # Define all 6 initialization experiments
    experiments = {
        'alpha0.75': {
            'method': 'biased',
            'params': {'alpha': 0.75},
            'checkpoint': 'results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.75/checkpoints/checkpoint-100.pt'
        },
        'alpha0.9': {
            'method': 'biased',
            'params': {'alpha': 0.9},
            'checkpoint': 'results/vgg16/cifar10/curves/initialization/biased_linear/alpha_0.9/checkpoints/checkpoint-100.pt'
        },
        'perturbed_small': {
            'method': 'perturbed',
            'params': {'alpha': 0.5, 'noise': 0.01},
            'checkpoint': 'results/vgg16/cifar10/curves/initialization/perturbed/noise_0.01/checkpoints/checkpoint-100.pt'
        },
        'perturbed_large': {
            'method': 'perturbed',
            'params': {'alpha': 0.5, 'noise': 0.1},
            'checkpoint': 'results/vgg16/cifar10/curves/initialization/perturbed/noise_0.1/checkpoints/checkpoint-100.pt'
        },
        'sphere_inside': {
            'method': 'sphere',
            'params': {'alpha': 0.5, 'noise': 0.01, 'inside': True},
            'checkpoint': 'results/vgg16/cifar10/curves/initialization/sphere_constrained/inside/checkpoints/checkpoint-100.pt'
        },
        'sphere_outside': {
            'method': 'sphere',
            'params': {'alpha': 0.5, 'noise': 0.01, 'inside': False},
            'checkpoint': 'results/vgg16/cifar10/curves/initialization/sphere_constrained/outside/checkpoints/checkpoint-100.pt'
        },
    }

    print("="*70)
    print("PART 1: INITIAL MIDDLE POINT L2 NORMS")
    print("="*70)
    print()

    # Store results
    initial_raw_l2_norms = {}
    initial_interpolated_l2_norms = {}

    # Recreate all initial middle points and calculate L2 norms
    for name, config in experiments.items():
        print(f"Recreating: {name}")
        print(f"  Method: {config['method']}")
        print(f"  Params: {config['params']}")

        middle_point, interpolated_l2 = recreate_initial_middle_point(
            endpoint0_path,
            endpoint1_path,
            init_method=config['method'],
            init_params=config['params'],
            seed=1
        )

        raw_l2_norm = torch.norm(middle_point).item()
        initial_raw_l2_norms[name] = raw_l2_norm
        initial_interpolated_l2_norms[name] = interpolated_l2

        print(f"  Shape: {middle_point.shape}")
        print(f"  Raw middle point L2 norm: {raw_l2_norm:.2f}")
        print(f"  Interpolated L2 norm (t=0.5): {interpolated_l2:.2f}")
        print()

    # Extract trained middle points and calculate L2 norms
    print("="*70)
    print("PART 2: TRAINED MIDDLE POINT L2 NORMS (after 100 epochs)")
    print("="*70)
    print()

    trained_l2_norms = {}

    for name, config in experiments.items():
        checkpoint_path = config['checkpoint']
        print(f"Loading: {name}")
        print(f"  Checkpoint: {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  WARNING: Checkpoint not found, skipping")
            print()
            continue

        middle_point = extract_middle_point_from_checkpoint(checkpoint_path)
        l2_norm = torch.norm(middle_point).item()
        trained_l2_norms[name] = l2_norm

        print(f"  Shape: {middle_point.shape}")
        print(f"  Trained L2 norm: {l2_norm:.2f}")
        print()

    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    print("RAW MIDDLE POINT L2 Norms (||w₁||):")
    print(f"{'Experiment':<20} {'Initial L2':>12} {'Trained L2':>12} {'Change':>12}")
    print("-" * 60)

    for name in experiments.keys():
        initial = initial_raw_l2_norms.get(name, float('nan'))
        trained = trained_l2_norms.get(name, float('nan'))
        change = trained - initial if name in trained_l2_norms else float('nan')

        if name in trained_l2_norms:
            print(f"{name:<20} {initial:>12.2f} {trained:>12.2f} {change:>+12.2f}")
        else:
            print(f"{name:<20} {initial:>12.2f} {'N/A':>12} {'N/A':>12}")

    print()
    print("INTERPOLATED L2 Norms at t=0.5:")
    print(f"{'Experiment':<20} {'Initial L2':>12}")
    print("-" * 35)

    for name in experiments.keys():
        interpolated = initial_interpolated_l2_norms.get(name, float('nan'))
        print(f"{name:<20} {interpolated:>12.2f}")

    print()
    print("INITIAL RAW L2 Statistics:")
    print(f"  Minimum: {min(initial_raw_l2_norms.values()):.2f} ({min(initial_raw_l2_norms, key=initial_raw_l2_norms.get)})")
    print(f"  Maximum: {max(initial_raw_l2_norms.values()):.2f} ({max(initial_raw_l2_norms, key=initial_raw_l2_norms.get)})")
    print(f"  Mean:    {np.mean(list(initial_raw_l2_norms.values())):.2f}")
    print(f"  Std:     {np.std(list(initial_raw_l2_norms.values())):.2f}")

    print()
    print("INITIAL INTERPOLATED L2 Statistics:")
    print(f"  Minimum: {min(initial_interpolated_l2_norms.values()):.2f} ({min(initial_interpolated_l2_norms, key=initial_interpolated_l2_norms.get)})")
    print(f"  Maximum: {max(initial_interpolated_l2_norms.values()):.2f} ({max(initial_interpolated_l2_norms, key=initial_interpolated_l2_norms.get)})")
    print(f"  Mean:    {np.mean(list(initial_interpolated_l2_norms.values())):.2f}")
    print(f"  Std:     {np.std(list(initial_interpolated_l2_norms.values())):.2f}")

    if trained_l2_norms:
        print()
        print("TRAINED Statistics:")
        print(f"  Minimum: {min(trained_l2_norms.values()):.2f} ({min(trained_l2_norms, key=trained_l2_norms.get)})")
        print(f"  Maximum: {max(trained_l2_norms.values()):.2f} ({max(trained_l2_norms, key=trained_l2_norms.get)})")
        print(f"  Mean:    {np.mean(list(trained_l2_norms.values())):.2f}")
        print(f"  Std:     {np.std(list(trained_l2_norms.values())):.2f}")

    print()
    print("="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    main()
