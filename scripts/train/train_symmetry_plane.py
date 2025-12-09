"""
Symmetry Plane Optimization via Projected Gradient Descent

Finds optimal point θ* on the symmetry plane between two trained models
such that the two-segment linear path w₁ → θ* → w₂ has minimal maximum loss.

Mathematical formulation:
- Symmetry plane: n · (θ - m) = 0, where n = w₂ - w₁, m = (w₁ + w₂) / 2
- Path segments: w₁ → θ* and θ* → w₂ (both linear)
- Objective: minimize max_{t∈[0,1]} L(w(t)) over both segments

Method: Projected Gradient Descent
1. Compute gradient of objective w.r.t. θ
2. Take unconstrained gradient step: θ' = θ - η∇J(θ)
3. Project back to plane: θ_new = θ' - ((θ' - m)·n / ||n||²) * n
"""

import argparse
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
import tabulate

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')

import data
import models
import utils


def compute_path_loss(model, w_start, w_end, t_values, loaders, criterion, device):
    """
    Compute loss at multiple points along a linear path segment.

    Args:
        model: Neural network model
        w_start: Starting weights (state dict)
        w_end: Ending weights (state dict)
        t_values: List of interpolation parameters in [0, 1]
        loaders: Data loaders
        criterion: Loss function
        device: torch device

    Returns:
        losses: Array of losses at each t value
        errors: Array of errors at each t value
    """
    losses = []
    errors = []

    for t in t_values:
        # Interpolate weights
        w_t = {}
        for key in w_start.keys():
            w_t[key] = (1 - t) * w_start[key] + t * w_end[key]

        # Load interpolated weights
        model.load_state_dict(w_t)

        # Evaluate
        test_res = utils.test(loaders['test'], model, criterion, device=device)
        losses.append(test_res['loss'])
        errors.append(test_res['accuracy'])

    return np.array(losses), 100.0 - np.array(errors)


def project_to_symmetry_plane(theta, midpoint, normal):
    """
    Project point theta onto the symmetry plane.

    The plane is defined by: n · (θ - m) = 0
    Projection removes the component parallel to normal vector.

    Args:
        theta: Current point (state dict)
        midpoint: Point on plane (state dict)
        normal: Normal vector to plane (state dict)

    Returns:
        projected_theta: Projected point (state dict)
    """
    # Compute displacement from midpoint
    displacement = {key: theta[key] - midpoint[key] for key in theta.keys()}

    # Compute dot product (displacement · normal)
    dot_product = sum(torch.sum(displacement[key] * normal[key])
                     for key in displacement.keys())

    # Compute ||normal||²
    normal_norm_sq = sum(torch.sum(normal[key] ** 2) for key in normal.keys())

    # Compute parallel component: ((θ - m)·n / ||n||²) * n
    scale = dot_product / normal_norm_sq
    parallel_component = {key: scale * normal[key] for key in normal.keys()}

    # Project: θ_new = θ - parallel_component
    projected = {key: theta[key] - parallel_component[key] for key in theta.keys()}

    return projected


def verify_on_plane(theta, midpoint, normal, tolerance=1e-6):
    """Verify that theta satisfies the plane constraint: n · (θ - m) = 0"""
    displacement = {key: theta[key] - midpoint[key] for key in theta.keys()}
    dot_product = sum(torch.sum(displacement[key] * normal[key])
                     for key in displacement.keys())
    return abs(dot_product.item()) < tolerance


def initialize_theta(mode, w1, w2, device):
    """
    Initialize θ on the symmetry plane.

    Args:
        mode: "midpoint" or "random"
        w1: First endpoint (state dict)
        w2: Second endpoint (state dict)
        device: torch device

    Returns:
        theta: Initial point on plane (state dict)
    """
    if mode == "midpoint":
        # Simple: start at midpoint
        theta = {key: 0.5 * (w1[key] + w2[key]) for key in w1.keys()}
    elif mode == "random":
        # Random point on plane
        # Strategy: midpoint + random vector perpendicular to normal
        midpoint = {key: 0.5 * (w1[key] + w2[key]) for key in w1.keys()}
        normal = {key: w2[key] - w1[key] for key in w1.keys()}

        # Generate random vector
        random_vec = {}
        for key in w1.keys():
            random_vec[key] = torch.randn_like(w1[key])

        # Project random vector to be perpendicular to normal
        # v_perp = v - (v·n / ||n||²) * n
        dot_vn = sum(torch.sum(random_vec[key] * normal[key]) for key in random_vec.keys())
        normal_norm_sq = sum(torch.sum(normal[key] ** 2) for key in normal.keys())
        scale = dot_vn / normal_norm_sq

        v_perp = {key: random_vec[key] - scale * normal[key] for key in random_vec.keys()}

        # Normalize perpendicular vector and scale by distance between endpoints
        v_norm = torch.sqrt(sum(torch.sum(v_perp[key] ** 2) for key in v_perp.keys()))
        endpoint_dist = torch.sqrt(normal_norm_sq)
        scale_factor = 0.1 * endpoint_dist / v_norm  # 10% of endpoint distance

        # θ = midpoint + scaled perpendicular vector
        theta = {key: midpoint[key] + scale_factor * v_perp[key] for key in midpoint.keys()}
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")

    return theta


def train_symmetry_plane(args):
    """Main training loop for symmetry plane optimization."""

    os.makedirs(args.dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 80)
    print("SYMMETRY PLANE OPTIMIZATION VIA PROJECTED GRADIENT DESCENT")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Initialization: {args.init_mode}")
    print(f"Optimization steps: {args.steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Eval points per segment: {args.eval_points}")
    print("=" * 80)

    # Load data
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        split_classes=None
    )

    # Load model architecture
    architecture = getattr(models, args.model)
    model = architecture.base(num_classes=num_classes, **architecture.kwargs)
    model.to(device)

    criterion = F.cross_entropy

    # Load endpoint checkpoints
    print("\nLoading endpoint checkpoints...")
    print(f"  Endpoint 1: {args.init_start}")
    print(f"  Endpoint 2: {args.init_end}")

    ckpt1 = torch.load(args.init_start, map_location=device)
    ckpt2 = torch.load(args.init_end, map_location=device)

    w1 = ckpt1.get('model_state', ckpt1)
    w2 = ckpt2.get('model_state', ckpt2)

    # Compute plane parameters
    print("\nComputing symmetry plane parameters...")
    midpoint = {key: 0.5 * (w1[key] + w2[key]) for key in w1.keys()}
    normal = {key: w2[key] - w1[key] for key in w1.keys()}

    # Compute L2 distance between endpoints
    l2_dist = torch.sqrt(sum(torch.sum(normal[key] ** 2) for key in normal.keys()))
    print(f"  L2 distance between endpoints: {l2_dist.item():.4f}")

    # Initialize θ on the plane
    print(f"\nInitializing θ using '{args.init_mode}' strategy...")
    theta = initialize_theta(args.init_mode, w1, w2, device)

    # Verify initialization is on plane
    assert verify_on_plane(theta, midpoint, normal), "Initialization not on plane!"
    print("  ✓ Verified θ is on symmetry plane")

    # Make theta parameters that require gradients
    theta_params = {key: val.clone().requires_grad_(True) for key, val in theta.items()}

    # Setup optimizer for theta
    optimizer = torch.optim.SGD([theta_params[key] for key in theta_params.keys()],
                                lr=args.lr, momentum=args.momentum)

    # Learning rate schedule (same as curve training)
    def lr_schedule(step):
        alpha = step / args.steps
        if alpha <= 0.5:
            return 1.0
        elif alpha <= 0.9:
            return 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            return 0.01

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Evaluation points along segments
    t_values = np.linspace(0, 1, args.eval_points + 2)[1:-1]  # Exclude endpoints

    print(f"\nStarting optimization for {args.steps} steps...")
    print("=" * 80)

    optimization_log = []
    best_max_loss = float('inf')
    best_theta = None

    for step in range(args.steps):
        step_start_time = time.time()

        # Evaluate current path
        # Segment 1: w1 → θ
        losses1, errors1 = compute_path_loss(model, w1, theta_params, t_values,
                                            loaders, criterion, device)

        # Segment 2: θ → w2
        losses2, errors2 = compute_path_loss(model, theta_params, w2, t_values,
                                            loaders, criterion, device)

        # Objective: max loss along path
        max_loss1 = np.max(losses1)
        max_loss2 = np.max(losses2)
        current_max_loss = max(max_loss1, max_loss2)

        # Average metrics for logging
        avg_loss = (np.mean(losses1) + np.mean(losses2)) / 2
        avg_error = (np.mean(errors1) + np.mean(errors2)) / 2

        # Compute gradient
        # We want to minimize the max loss, so we backprop through the worst point
        optimizer.zero_grad()

        if max_loss1 >= max_loss2:
            # Worst point is on segment 1
            worst_t_idx = np.argmax(losses1)
            worst_t = t_values[worst_t_idx]

            # Interpolate to worst point
            w_t = {}
            for key in w1.keys():
                w_t[key] = (1 - worst_t) * w1[key] + worst_t * theta_params[key]
            model.load_state_dict(w_t)

            # Forward pass to compute loss
            loss_sum = 0.0
            for input, target in loaders['train']:
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss_sum += criterion(output, target)

            loss = loss_sum / len(loaders['train'])
            loss.backward()
        else:
            # Worst point is on segment 2
            worst_t_idx = np.argmax(losses2)
            worst_t = t_values[worst_t_idx]

            # Interpolate to worst point
            w_t = {}
            for key in theta_params.keys():
                w_t[key] = (1 - worst_t) * theta_params[key] + worst_t * w2[key]
            model.load_state_dict(w_t)

            # Forward pass to compute loss
            loss_sum = 0.0
            for input, target in loaders['train']:
                input, target = input.to(device), target.to(device)
                output = model(input)
                loss_sum += criterion(output, target)

            loss = loss_sum / len(loaders['train'])
            loss.backward()

        # Gradient step (unconstrained)
        optimizer.step()

        # Project back to symmetry plane
        with torch.no_grad():
            theta_params = project_to_symmetry_plane(theta_params, midpoint, normal)

            # Verify projection
            assert verify_on_plane(theta_params, midpoint, normal), \
                f"Projection failed at step {step}!"

        # Update learning rate
        scheduler.step()

        # Track best
        if current_max_loss < best_max_loss:
            best_max_loss = current_max_loss
            best_theta = {key: val.detach().clone() for key, val in theta_params.items()}

        step_time = time.time() - step_start_time

        # Log
        log_entry = {
            'step': step,
            'max_loss': current_max_loss,
            'avg_loss': avg_loss,
            'avg_error': avg_error,
            'lr': scheduler.get_last_lr()[0],
            'time': step_time
        }
        optimization_log.append(log_entry)

        # Print progress
        if step % args.print_freq == 0 or step == args.steps - 1:
            print(f"Step {step:4d}/{args.steps} | "
                  f"Max Loss: {current_max_loss:.4f} | "
                  f"Avg Loss: {avg_loss:.4f} | "
                  f"Avg Err: {avg_error:.2f}% | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                  f"Time: {step_time:.2f}s")

    print("=" * 80)
    print(f"Optimization complete!")
    print(f"Best max loss: {best_max_loss:.4f}")

    # Save optimized theta
    save_path = os.path.join(args.dir, 'checkpoint_optimal.pt')
    torch.save({'model_state': best_theta}, save_path)
    print(f"\nSaved optimal θ* to: {save_path}")

    # Save optimization log
    log_path = os.path.join(args.dir, 'optimization_log.json')
    import json
    with open(log_path, 'w') as f:
        json.dump(optimization_log, f, indent=2)
    print(f"Saved optimization log to: {log_path}")

    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Symmetry Plane Optimization')

    # Endpoints
    parser.add_argument('--init_start', type=str, required=True,
                       help='Path to first endpoint checkpoint')
    parser.add_argument('--init_end', type=str, required=True,
                       help='Path to second endpoint checkpoint')

    # Model and data
    parser.add_argument('--model', type=str, required=True, help='Model architecture')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--data_path', type=str, required=True, help='Path to dataset')
    parser.add_argument('--transform', type=str, required=True, help='Transform type')
    parser.add_argument('--use_test', action='store_true', help='Use test set')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')

    # Optimization
    parser.add_argument('--steps', type=int, default=200, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.015, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--init_mode', type=str, default='midpoint',
                       choices=['midpoint', 'random'], help='Initialization strategy')
    parser.add_argument('--eval_points', type=int, default=30,
                       help='Number of evaluation points per segment')

    # Output
    parser.add_argument('--dir', type=str, required=True, help='Output directory')
    parser.add_argument('--print_freq', type=int, default=10, help='Print frequency')

    args = parser.parse_args()

    train_symmetry_plane(args)
