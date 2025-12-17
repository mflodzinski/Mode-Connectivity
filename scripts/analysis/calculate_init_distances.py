"""Calculate L2 distances between initial middle control points of different initialization methods.

This script recreates the initial middle control points for 6 different initialization methods
and calculates pairwise L2 distances to analyze how different the initializations are.
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
scripts_root = os.path.join(script_dir, '..')
sys.path.insert(0, scripts_root)
external_path = os.path.join(scripts_root, '..', 'external', 'dnn-mode-connectivity')
sys.path.insert(0, external_path)

import curves as curve_module
import models as model_module

def load_endpoint(checkpoint_path):
    """Load endpoint model state dict."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint['model_state']

def create_curve_model(num_classes=10, architecture='VGG16'):
    """Create a Bezier curve model."""
    arch = getattr(model_module, architecture)

    model = curve_module.CurveNet(
        num_classes=num_classes,
        curve=curve_module.Bezier,
        architecture=arch.curve,
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
    model.import_base_parameters(base_model, 0)

    # Load endpoint 1
    base_model.load_state_dict(endpoint1_state)
    model.import_base_parameters(base_model, 2)

def extract_middle_point(model):
    """Extract middle control point (bend 1) from curve model."""
    middle_params = []

    for name, param in model.named_parameters():
        if '_1' in name:  # Middle bend has suffix '_1'
            middle_params.append(param.data.flatten())

    # Concatenate all parameters into single vector
    middle_vector = torch.cat(middle_params)
    return middle_vector

def extract_middle_point_from_checkpoint(checkpoint_path):
    """Extract middle control point directly from a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        torch.Tensor: Flattened middle control point parameters
    """
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

    Args:
        endpoint0_path: Path to endpoint 0 checkpoint
        endpoint1_path: Path to endpoint 1 checkpoint
        init_method: Initialization method name ('linear', 'biased', 'perturbed', 'sphere')
        init_params: Dict of parameters for initialization method
        seed: Random seed for stochastic methods

    Returns:
        torch.Tensor: Flattened middle control point parameters
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

    # Apply initialization method
    if init_method == 'linear':
        model.init_linear()
    elif init_method == 'biased':
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

    # Extract middle point
    middle_point = extract_middle_point(model)

    return middle_point

def calculate_l2_distance(point1, point2):
    """Calculate L2 distance between two parameter vectors."""
    return torch.norm(point1 - point2).item()

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
    print("PART 1: RECREATING INITIAL MIDDLE CONTROL POINTS")
    print("="*70)
    print()

    # Recreate all initial middle points
    initial_middle_points = {}
    for name, config in experiments.items():
        print(f"Recreating: {name} ({config['method']} initialization)")
        middle_point = recreate_initial_middle_point(
            endpoint0_path,
            endpoint1_path,
            init_method=config['method'],
            init_params=config['params'],
            seed=1
        )
        initial_middle_points[name] = middle_point
        print(f"  Shape: {middle_point.shape}")
        print(f"  L2 norm: {torch.norm(middle_point).item():.2f}")
        print()

    # Extract trained middle points from checkpoints
    print("="*70)
    print("PART 2: EXTRACTING TRAINED MIDDLE CONTROL POINTS")
    print("="*70)
    print()

    trained_middle_points = {}
    for name, config in experiments.items():
        checkpoint_path = config['checkpoint']
        print(f"Loading: {name} from {checkpoint_path}")

        if not os.path.exists(checkpoint_path):
            print(f"  ⚠️  WARNING: Checkpoint not found, skipping")
            print()
            continue

        middle_point = extract_middle_point_from_checkpoint(checkpoint_path)
        trained_middle_points[name] = middle_point
        print(f"  Shape: {middle_point.shape}")
        print(f"  L2 norm: {torch.norm(middle_point).item():.2f}")
        print()

    # Calculate pairwise distances for INITIAL points
    print("="*70)
    print("PART 3: CALCULATING PAIRWISE L2 DISTANCES (INITIAL POINTS)")
    print("="*70)
    print()

    names = list(initial_middle_points.keys())
    n = len(names)
    initial_distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                initial_distance_matrix[i, j] = 0.0
            else:
                dist = calculate_l2_distance(initial_middle_points[names[i]], initial_middle_points[names[j]])
                initial_distance_matrix[i, j] = dist

    # Calculate pairwise distances for TRAINED points
    print("="*70)
    print("PART 4: CALCULATING PAIRWISE L2 DISTANCES (TRAINED POINTS)")
    print("="*70)
    print()

    trained_distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == j:
                trained_distance_matrix[i, j] = 0.0
            else:
                if names[i] in trained_middle_points and names[j] in trained_middle_points:
                    dist = calculate_l2_distance(trained_middle_points[names[i]], trained_middle_points[names[j]])
                    trained_distance_matrix[i, j] = dist
                else:
                    trained_distance_matrix[i, j] = np.nan  # Mark as missing

    # Print INITIAL distance matrix
    print("INITIAL Distance Matrix:")
    print()
    header = "".ljust(20) + "".join([name[:15].ljust(15) for name in names])
    print(header)
    print("-" * len(header))
    for i, name_i in enumerate(names):
        row = name_i[:20].ljust(20)
        for j in range(n):
            row += f"{initial_distance_matrix[i, j]:14.2f} "
        print(row)
    print()

    # Print TRAINED distance matrix
    print("TRAINED Distance Matrix:")
    print()
    print(header)
    print("-" * len(header))
    for i, name_i in enumerate(names):
        row = name_i[:20].ljust(20)
        for j in range(n):
            val = trained_distance_matrix[i, j]
            if np.isnan(val):
                row += f"{'N/A':>14} "
            else:
                row += f"{val:14.2f} "
        print(row)
    print()

    # Print summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print()

    # Get upper triangle (exclude diagonal) for INITIAL
    initial_upper_triangle = initial_distance_matrix[np.triu_indices(n, k=1)]

    print("INITIAL Middle Points:")
    print(f"  Number of pairs: {len(initial_upper_triangle)}")
    print(f"  Mean distance: {np.mean(initial_upper_triangle):.2f}")
    print(f"  Std distance: {np.std(initial_upper_triangle):.2f}")
    print(f"  Min distance: {np.min(initial_upper_triangle):.2f}")
    print(f"  Max distance: {np.max(initial_upper_triangle):.2f}")
    print()

    # Find most similar and most different pairs for INITIAL
    initial_min_idx = np.unravel_index(np.argmin(initial_distance_matrix + np.eye(n) * 1e10), (n, n))
    initial_max_idx = np.unravel_index(np.argmax(initial_distance_matrix), (n, n))

    print(f"  Most similar pair:")
    print(f"    {names[initial_min_idx[0]]} <-> {names[initial_min_idx[1]]}")
    print(f"    Distance: {initial_distance_matrix[initial_min_idx]:.2f}")
    print()

    print(f"  Most different pair:")
    print(f"    {names[initial_max_idx[0]]} <-> {names[initial_max_idx[1]]}")
    print(f"    Distance: {initial_distance_matrix[initial_max_idx]:.2f}")
    print()

    # Get upper triangle (exclude diagonal) for TRAINED
    trained_upper_triangle = trained_distance_matrix[np.triu_indices(n, k=1)]
    trained_upper_triangle = trained_upper_triangle[~np.isnan(trained_upper_triangle)]

    if len(trained_upper_triangle) > 0:
        print("TRAINED Middle Points:")
        print(f"  Number of pairs: {len(trained_upper_triangle)}")
        print(f"  Mean distance: {np.mean(trained_upper_triangle):.2f}")
        print(f"  Std distance: {np.std(trained_upper_triangle):.2f}")
        print(f"  Min distance: {np.min(trained_upper_triangle):.2f}")
        print(f"  Max distance: {np.max(trained_upper_triangle):.2f}")
        print()

        # Find most similar and most different pairs for TRAINED
        trained_min_idx = np.unravel_index(np.argmin(trained_distance_matrix + np.eye(n) * 1e10), (n, n))
        trained_max_idx = np.unravel_index(np.argmax(trained_distance_matrix), (n, n))

        print(f"  Most similar pair:")
        print(f"    {names[trained_min_idx[0]]} <-> {names[trained_min_idx[1]]}")
        print(f"    Distance: {trained_distance_matrix[trained_min_idx]:.2f}")
        print()

        print(f"  Most different pair:")
        print(f"    {names[trained_max_idx[0]]} <-> {names[trained_max_idx[1]]}")
        print(f"    Distance: {trained_distance_matrix[trained_max_idx]:.2f}")
        print()
    else:
        print("TRAINED Middle Points: No trained checkpoints found")
        print()

    # Save outputs
    output_dir = 'results/vgg16/cifar10/curves/initialization/analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Save both distance matrices as NumPy archive
    npz_path = os.path.join(output_dir, 'initialization_distances.npz')
    np.savez(
        npz_path,
        initial_distance_matrix=initial_distance_matrix,
        trained_distance_matrix=trained_distance_matrix,
        names=names,
        initial_mean_distance=np.mean(initial_upper_triangle),
        initial_std_distance=np.std(initial_upper_triangle),
        initial_min_distance=np.min(initial_upper_triangle),
        initial_max_distance=np.max(initial_upper_triangle),
        trained_mean_distance=np.mean(trained_upper_triangle),
        trained_std_distance=np.std(trained_upper_triangle),
        trained_min_distance=np.min(trained_upper_triangle),
        trained_max_distance=np.max(trained_upper_triangle)
    )
    print(f"✓ Saved distance matrices to: {npz_path}")

    # Save summary as text
    txt_path = os.path.join(output_dir, 'initialization_distances_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("L2 DISTANCES BETWEEN MIDDLE CONTROL POINTS\n")
        f.write("="*70 + "\n\n")

        # INITIAL DISTANCES
        f.write("="*70 + "\n")
        f.write("PART 1: INITIAL MIDDLE CONTROL POINTS\n")
        f.write("="*70 + "\n\n")

        f.write("Distance Matrix:\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, name_i in enumerate(names):
            row = name_i[:20].ljust(20)
            for j in range(n):
                row += f"{initial_distance_matrix[i, j]:14.2f} "
            f.write(row + "\n")
        f.write("\n")

        f.write("Summary Statistics:\n")
        f.write(f"  Number of pairs: {len(initial_upper_triangle)}\n")
        f.write(f"  Mean distance: {np.mean(initial_upper_triangle):.2f}\n")
        f.write(f"  Std distance: {np.std(initial_upper_triangle):.2f}\n")
        f.write(f"  Min distance: {np.min(initial_upper_triangle):.2f}\n")
        f.write(f"  Max distance: {np.max(initial_upper_triangle):.2f}\n\n")

        f.write(f"Most similar pair:\n")
        f.write(f"  {names[initial_min_idx[0]]} <-> {names[initial_min_idx[1]]}\n")
        f.write(f"  Distance: {initial_distance_matrix[initial_min_idx]:.2f}\n\n")

        f.write(f"Most different pair:\n")
        f.write(f"  {names[initial_max_idx[0]]} <-> {names[initial_max_idx[1]]}\n")
        f.write(f"  Distance: {initial_distance_matrix[initial_max_idx]:.2f}\n\n")

        # TRAINED DISTANCES
        f.write("="*70 + "\n")
        f.write("PART 2: TRAINED MIDDLE CONTROL POINTS (after 100 epochs)\n")
        f.write("="*70 + "\n\n")

        f.write("Distance Matrix:\n\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for i, name_i in enumerate(names):
            row = name_i[:20].ljust(20)
            for j in range(n):
                row += f"{trained_distance_matrix[i, j]:14.2f} "
            f.write(row + "\n")
        f.write("\n")

        f.write("Summary Statistics:\n")
        f.write(f"  Number of pairs: {len(trained_upper_triangle)}\n")
        f.write(f"  Mean distance: {np.mean(trained_upper_triangle):.2f}\n")
        f.write(f"  Std distance: {np.std(trained_upper_triangle):.2f}\n")
        f.write(f"  Min distance: {np.min(trained_upper_triangle):.2f}\n")
        f.write(f"  Max distance: {np.max(trained_upper_triangle):.2f}\n\n")

        f.write(f"Most similar pair:\n")
        f.write(f"  {names[trained_min_idx[0]]} <-> {names[trained_min_idx[1]]}\n")
        f.write(f"  Distance: {trained_distance_matrix[trained_min_idx]:.2f}\n\n")

        f.write(f"Most different pair:\n")
        f.write(f"  {names[trained_max_idx[0]]} <-> {names[trained_max_idx[1]]}\n")
        f.write(f"  Distance: {trained_distance_matrix[trained_max_idx]:.2f}\n\n")

        # COMPARISON
        f.write("="*70 + "\n")
        f.write("COMPARISON: INITIAL vs TRAINED\n")
        f.write("="*70 + "\n\n")

        f.write("Mean distances:\n")
        f.write(f"  Initial: {np.mean(initial_upper_triangle):.2f}\n")
        f.write(f"  Trained: {np.mean(trained_upper_triangle):.2f}\n")
        f.write(f"  Change:  {np.mean(trained_upper_triangle) - np.mean(initial_upper_triangle):.2f}\n\n")

        f.write("Standard deviation:\n")
        f.write(f"  Initial: {np.std(initial_upper_triangle):.2f}\n")
        f.write(f"  Trained: {np.std(trained_upper_triangle):.2f}\n")
        f.write(f"  Change:  {np.std(trained_upper_triangle) - np.std(initial_upper_triangle):.2f}\n\n")

    print(f"✓ Saved summary to: {txt_path}")

    # Create heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Initial distances heatmap
    sns.heatmap(
        initial_distance_matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt='.1f',
        cmap='viridis',
        cbar_kws={'label': 'L2 Distance'},
        ax=axes[0]
    )
    axes[0].set_title('Initial Middle Control Points', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Initialization Method', fontsize=12)
    axes[0].set_ylabel('Initialization Method', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)

    # Trained distances heatmap
    sns.heatmap(
        trained_distance_matrix,
        xticklabels=names,
        yticklabels=names,
        annot=True,
        fmt='.1f',
        cmap='viridis',
        cbar_kws={'label': 'L2 Distance'},
        ax=axes[1]
    )
    axes[1].set_title('Trained Middle Control Points (100 epochs)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Initialization Method', fontsize=12)
    axes[1].set_ylabel('Initialization Method', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)

    plt.suptitle('L2 Distances Between Middle Control Points', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    heatmap_path = os.path.join(output_dir, 'initialization_distances_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved heatmap to: {heatmap_path}")

    print()
    print("="*70)
    print("DONE")
    print("="*70)

if __name__ == "__main__":
    main()
