"""
Pure XOR Experiment: Test if networks computing the same function lie in the same basin.

This experiment:
1. Trains N networks on the 4-point XOR problem from different random initializations
2. All networks achieve 100% accuracy (compute identical functions)
3. For each pair, measures the barrier before and after permutation alignment
4. Tests if the one-basin hypothesis holds for functionally equivalent networks

Architecture: 2-H-1 MLP (2 inputs, H hidden neurons, 1 output for binary classification)
             H is configurable via --hidden-neurons (default: 4)
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from itertools import combinations, permutations
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from mode_connectivity.evaluation.metrics import state_distance_summary

# =============================================================================
# XOR Dataset
# =============================================================================
XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


# =============================================================================
# Configurable MLP
# =============================================================================
class SimpleMLP(nn.Module):
    """Simple MLP for XOR: 2 inputs -> H hidden -> 1 output (logit).

    Args:
        hidden_size: Number of hidden neurons (default: 4)
    """

    def __init__(self, hidden_size=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(2, hidden_size)  # Input to hidden
        self.fc2 = nn.Linear(hidden_size, 1)  # Hidden to output (binary logit)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def get_training_config(hidden_size):
    """Get training hyperparameters based on hidden layer size.

    Args:
        hidden_size: Number of hidden neurons

    Returns:
        Dict with max_epochs, lr, init_min, init_max
    """
    return {
        'max_epochs': 5000,
        'lr': 0.05,
        'init_min': -0.5,
        'init_max': 0.5,
    }




# =============================================================================
# Training
# =============================================================================
def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_xor_network(seed, hidden_size=4, max_epochs=None, lr=None, verbose=False,
                      loss_threshold=1e-4, patience=500, lr_patience=200, lr_factor=0.5,
                      track_config=None):
    """Train a network on XOR until loss converges.

    Convergence is defined as loss not decreasing meaningfully for `patience` epochs.

    Args:
        seed: Random seed for initialization
        hidden_size: Number of hidden neurons
        max_epochs: Maximum training epochs (None = use default for hidden_size)
        lr: Learning rate (None = use default for hidden_size)
        verbose: Print training progress
        loss_threshold: Minimum loss improvement to count as progress
        patience: Number of epochs without improvement before stopping
        lr_patience: Epochs without improvement before reducing LR
        lr_factor: Multiplicative LR drop on plateau
        track_config: Optional dict for boundary tracking

    Returns:
        Trained model, final_loss, final_accuracy, stop_reason, epochs_trained
    """
    set_seed(seed)

    # Get default hyperparameters based on hidden size
    config = get_training_config(hidden_size)
    if max_epochs is None:
        max_epochs = config['max_epochs']
    else:
        max_epochs = min(max_epochs, config['max_epochs'])
    if lr is None:
        lr = config['lr']
    model = SimpleMLP(hidden_size=hidden_size)

    # Small random initialization to break symmetry
    with torch.no_grad():
        for param in model.parameters():
            nn.init.uniform_(param, a=config['init_min'], b=config['init_max'])

    # Full-batch gradient descent (batch size = 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=lr_factor,
        patience=lr_patience,
        threshold=loss_threshold,
        threshold_mode='abs',
        min_lr=5e-4,
    )

    best_loss = float('inf')
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(XOR_DATA)
        loss = F.binary_cross_entropy_with_logits(outputs, XOR_LABELS)

        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        scheduler.step(current_loss)

        if track_config is not None:
            interval = track_config['interval']
            if (epoch + 1) % interval == 0:
                xx, yy, Z = compute_boundary_grid(model, grid_resolution=track_config['grid_resolution'])
                acc = evaluate_model(model)['accuracy']
                track_config['frames'].append({'epoch': epoch + 1, 'Z': Z, 'accuracy': acc})
                frame_path = os.path.join(track_config['frames_dir'], f'epoch_{epoch + 1:05d}.png')
                save_boundary_plot(xx, yy, Z, frame_path)
                track_config['last_epoch'] = epoch + 1

        # Check for convergence based on loss improvement
        if best_loss - current_loss > loss_threshold:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Converged if no improvement for patience epochs
        if epochs_without_improvement >= patience:
            with torch.no_grad():
                pred = (torch.sigmoid(model(XOR_DATA)) >= 0.5).long()
                accuracy = (pred == XOR_LABELS.long()).float().mean().item() * 100
            if track_config is not None and track_config.get('last_epoch') != (epoch + 1):
                xx, yy, Z = compute_boundary_grid(model, grid_resolution=track_config['grid_resolution'])
                acc = evaluate_model(model)['accuracy']
                track_config['frames'].append({'epoch': epoch + 1, 'Z': Z, 'accuracy': acc})
                frame_path = os.path.join(track_config['frames_dir'], f'epoch_{epoch + 1:05d}.png')
                save_boundary_plot(xx, yy, Z, frame_path)
                track_config['last_epoch'] = epoch + 1
            if verbose:
                print(f"  Seed {seed}: Converged at epoch {epoch + 1}, loss={current_loss:.6f}, acc={accuracy:.0f}%")
            return model, current_loss, accuracy, 'converged', epoch + 1

        if verbose and (epoch + 1) % 1000 == 0:
            with torch.no_grad():
                pred = (torch.sigmoid(outputs) >= 0.5).long()
                accuracy = (pred == XOR_LABELS.long()).float().mean().item()
            print(f"  Seed {seed}: Epoch {epoch + 1}, loss={current_loss:.6f}, acc={accuracy:.2f}")

    # Reached max epochs
    with torch.no_grad():
        pred = (torch.sigmoid(model(XOR_DATA)) >= 0.5).long()
        accuracy = (pred == XOR_LABELS.long()).float().mean().item() * 100
    if track_config is not None and track_config.get('last_epoch') != max_epochs:
        xx, yy, Z = compute_boundary_grid(model, grid_resolution=track_config['grid_resolution'])
        acc = evaluate_model(model)['accuracy']
        track_config['frames'].append({'epoch': max_epochs, 'Z': Z, 'accuracy': acc})
        frame_path = os.path.join(track_config['frames_dir'], f'epoch_{max_epochs:05d}.png')
        save_boundary_plot(xx, yy, Z, frame_path)
        track_config['last_epoch'] = max_epochs
    if verbose:
        print(f"  Seed {seed}: Max epochs reached, loss={current_loss:.6f}, acc={accuracy:.0f}%")
    return model, current_loss, accuracy, 'max_epochs', max_epochs


# =============================================================================
# Evaluation
# =============================================================================
def evaluate_model(model):
    """Evaluate model on XOR dataset."""
    model.eval()
    with torch.no_grad():
        outputs = model(XOR_DATA)
        loss = F.binary_cross_entropy_with_logits(outputs, XOR_LABELS).item()
        pred = (torch.sigmoid(outputs) >= 0.5).long()
        accuracy = (pred == XOR_LABELS.long()).float().mean().item() * 100
    return {'loss': loss, 'accuracy': accuracy}


def compute_l2_distance(state_a, state_b):
    """Compute L2 distance between two state dicts."""
    return state_distance_summary(state_a, state_b)


def vectorize_model_params(model):
    """Return all trainable parameters as a single 1D tensor."""
    return torch.cat([p.detach().view(-1) for p in model.parameters()])


def compute_barrier(model_a, model_b, num_points=21):
    """Compute linear interpolation barrier between two models.

    Args:
        model_a: First model
        model_b: Second model
        num_points: Number of interpolation points

    Returns:
        Dictionary with interpolation results
    """
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    # Infer hidden size from model weights
    hidden_size = state_a['fc1.weight'].shape[0]
    interp_model = SimpleMLP(hidden_size=hidden_size)

    ts = np.linspace(0, 1, num_points)
    results = {
        't': ts.tolist(),
        'loss': [],
        'accuracy': [],
    }

    for t in ts:
        # Linear interpolation
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1 - t) * state_a[key] + t * state_b[key]
        interp_model.load_state_dict(interp_state)

        # Evaluate
        res = evaluate_model(interp_model)
        results['loss'].append(res['loss'])
        results['accuracy'].append(res['accuracy'])

    # Compute barrier metrics
    results['min_accuracy'] = min(results['accuracy'])
    results['max_loss'] = max(results['loss'])
    results['barrier'] = 100 - results['min_accuracy']  # Error barrier
    results['endpoint_avg_loss'] = (results['loss'][0] + results['loss'][-1]) / 2
    results['loss_barrier'] = results['max_loss'] - results['endpoint_avg_loss']

    return results


# =============================================================================
# Alignment via Exhaustive Search
# =============================================================================
def apply_permutation_to_state(state, perm):
    """Apply a permutation to the hidden layer of a state dict.

    Args:
        state: Model state dict
        perm: List/array representing permutation of hidden neurons

    Returns:
        New state dict with permutation applied
    """
    perm = list(perm)
    new_state = OrderedDict()

    # fc1.weight: shape (hidden, input) - permute rows
    new_state['fc1.weight'] = state['fc1.weight'][perm, :]

    # fc1.bias: shape (hidden,) - permute elements
    new_state['fc1.bias'] = state['fc1.bias'][perm]

    # fc2.weight: shape (output, hidden) - permute columns
    new_state['fc2.weight'] = state['fc2.weight'][:, perm]

    # fc2.bias: shape (output,) - unchanged
    new_state['fc2.bias'] = state['fc2.bias'].clone()

    return new_state


def align_models_exhaustive(model_a, model_b, verbose=False):
    """Align model_b to model_a using exhaustive permutation search.

    Tries all possible permutations of hidden neurons and picks the one
    that minimizes L2 distance to model_a.

    Args:
        model_a: Reference model
        model_b: Model to align
        verbose: Print search progress

    Returns:
        Aligned model, best permutation found, all permutation results
    """
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    hidden_size = state_b['fc1.weight'].shape[0]
    all_perms = list(permutations(range(hidden_size)))

    if verbose:
        print(f"    Searching {len(all_perms)} permutations...")

    best_perm = None
    best_distance = float('inf')
    best_barrier = float('inf')
    best_loss_barrier = float('inf')
    perm_results = []

    for perm in all_perms:
        # Apply permutation
        permuted_state = apply_permutation_to_state(state_b, perm)

        # Compute L2 distance
        dist = compute_l2_distance(state_a, permuted_state)['l2_distance']

        # Also compute barrier for this permutation
        temp_model = SimpleMLP(hidden_size=hidden_size)
        temp_model.load_state_dict(permuted_state)
        barrier_result = compute_barrier(model_a, temp_model)
        barrier = barrier_result['barrier']
        loss_barrier = barrier_result['loss_barrier']

        perm_results.append({
            'perm': list(perm),
            'l2_distance': dist,
            'barrier': barrier,
            'loss_barrier': loss_barrier,
        })

        # Selection priority: error barrier -> loss barrier -> L2 distance
        better = False
        if barrier < best_barrier:
            better = True
        elif barrier == best_barrier:
            if loss_barrier < best_loss_barrier:
                better = True
            elif loss_barrier == best_loss_barrier and dist < best_distance:
                better = True

        if better:
            best_barrier = barrier
            best_loss_barrier = loss_barrier
            best_distance = dist
            best_perm = perm

    # Create aligned model with best permutation
    aligned_state = apply_permutation_to_state(state_b, best_perm)
    aligned_model = SimpleMLP(hidden_size=hidden_size)
    aligned_model.load_state_dict(aligned_state)

    return aligned_model, list(best_perm), perm_results


# =============================================================================
# Decision Boundary Visualization
# =============================================================================
def plot_confidence_boundary(ax, xx, yy, probs):
    """Plot confidence heatmap and 0.5 decision contour."""
    contour = ax.contourf(
        xx, yy, probs,
        levels=np.linspace(0.0, 1.0, 41),
        cmap='RdYlGn',
        vmin=0.0,
        vmax=1.0,
        alpha=0.85,
    )
    ax.contour(xx, yy, probs, levels=[0.5], colors=['black'], linewidths=2)
    return contour


def plot_decision_boundaries(models, output_path, grid_resolution=200):
    """Plot decision boundaries for all trained networks.

    Creates a grid plot showing each network's decision boundary along with
    the XOR data points.

    Args:
        models: Dict of seed -> model
        output_path: Path to save the plot
        grid_resolution: Resolution of the decision boundary grid
    """
    num_models = len(models)
    cols = min(3, num_models)
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if num_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Create meshgrid for decision boundary
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32
    )

    # Colors for XOR points
    colors = ['#d62728', '#2ca02c']  # Red for 0, Green for 1

    for idx, (seed, model) in enumerate(models.items()):
        ax = axes[idx]

        # Get predictions on grid
        model.eval()
        with torch.no_grad():
            outputs = model(grid_points)
            probs = torch.sigmoid(outputs).squeeze(1).numpy()

        # Reshape probabilities to grid
        Z = probs.reshape(xx.shape)

        # Plot confidence and decision boundary
        plot_confidence_boundary(ax, xx, yy, Z)

        # Plot XOR data points
        labels = XOR_LABELS.long().squeeze(1).numpy()
        for i, (point, label) in enumerate(zip(XOR_DATA.numpy(), labels)):
            ax.scatter(point[0], point[1], c=colors[label], s=200,
                       edgecolors='black', linewidths=2, zorder=5)
            ax.annotate(f'({int(point[0])},{int(point[1])})→{label}',
                        (point[0], point[1]), textcoords="offset points",
                        xytext=(10, 5), fontsize=9)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x₁', fontsize=11)
        ax.set_ylabel('x₂', fontsize=11)
        ax.set_title(f'Network (seed={seed})', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Decision boundaries saved to {output_path}")
    plt.close(fig)


def plot_interpolation_boundaries(model_a, model_b, seed_a, seed_b, output_path,
                                   num_steps=5, grid_resolution=150):
    """Plot decision boundaries along the interpolation path between two models.

    Args:
        model_a: First model
        model_b: Second model
        seed_a: Seed of first model
        seed_b: Seed of second model
        output_path: Path to save the plot
        num_steps: Number of interpolation steps to show
        grid_resolution: Resolution of the decision boundary grid
    """
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()

    ts = np.linspace(0, 1, num_steps)

    fig, axes = plt.subplots(1, num_steps, figsize=(4 * num_steps, 4))

    # Create meshgrid
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32
    )

    # Colors for XOR points
    colors = ['#d62728', '#2ca02c']

    # Infer hidden size from model weights
    hidden_size = state_a['fc1.weight'].shape[0]
    interp_model = SimpleMLP(hidden_size=hidden_size)

    for idx, t in enumerate(ts):
        ax = axes[idx]

        # Interpolate weights
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1 - t) * state_a[key] + t * state_b[key]
        interp_model.load_state_dict(interp_state)

        # Get predictions
        interp_model.eval()
        with torch.no_grad():
            outputs = interp_model(grid_points)
            probs = torch.sigmoid(outputs).squeeze(1).numpy()

        # Evaluate accuracy
        res = evaluate_model(interp_model)
        acc = res['accuracy']

        Z = probs.reshape(xx.shape)

        # Plot confidence and decision boundary
        plot_confidence_boundary(ax, xx, yy, Z)

        # Plot XOR data points
        labels = XOR_LABELS.long().squeeze(1).numpy()
        for point, label in zip(XOR_DATA.numpy(), labels):
            ax.scatter(point[0], point[1], c=colors[label], s=150,
                       edgecolors='black', linewidths=2, zorder=5)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('x₁', fontsize=10)
        if idx == 0:
            ax.set_ylabel('x₂', fontsize=10)
        ax.set_title(f't={t:.2f}\nAcc={acc:.0f}%', fontsize=10)
        ax.set_aspect('equal')

    plt.suptitle(f'Interpolation: seed{seed_a} → seed{seed_b}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Interpolation boundaries saved to {output_path}")
    plt.close(fig)


def plot_interpolation_boundaries_stacked(model_a, model_b_before, model_b_after,
                                          seed_a, seed_b, output_path,
                                          num_steps=5, grid_resolution=150):
    """Plot interpolation boundaries stacked: before (top) and after (bottom)."""
    state_a = model_a.state_dict()
    state_before = model_b_before.state_dict()
    state_after = model_b_after.state_dict()

    ts = np.linspace(0, 1, num_steps)
    fig, axes = plt.subplots(2, num_steps, figsize=(4 * num_steps, 8))

    if num_steps == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32
    )

    colors = ['#d62728', '#2ca02c']
    labels = XOR_LABELS.long().squeeze(1).numpy()

    hidden_size = state_a['fc1.weight'].shape[0]
    interp_before = SimpleMLP(hidden_size=hidden_size)
    interp_after = SimpleMLP(hidden_size=hidden_size)

    for idx, t in enumerate(ts):
        ax_before = axes[0, idx]
        ax_after = axes[1, idx]

        # Before alignment interpolation
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1 - t) * state_a[key] + t * state_before[key]
        interp_before.load_state_dict(interp_state)

        interp_before.eval()
        with torch.no_grad():
            outputs = interp_before(grid_points)
            probs = torch.sigmoid(outputs).squeeze(1).numpy()
        acc_before = evaluate_model(interp_before)['accuracy']
        Z_before = probs.reshape(xx.shape)

        plot_confidence_boundary(ax_before, xx, yy, Z_before)
        for point, label in zip(XOR_DATA.numpy(), labels):
            ax_before.scatter(point[0], point[1], c=colors[label], s=150,
                              edgecolors='black', linewidths=2, zorder=5)

        # After alignment interpolation
        interp_state = OrderedDict()
        for key in state_a:
            interp_state[key] = (1 - t) * state_a[key] + t * state_after[key]
        interp_after.load_state_dict(interp_state)

        interp_after.eval()
        with torch.no_grad():
            outputs = interp_after(grid_points)
            probs = torch.sigmoid(outputs).squeeze(1).numpy()
        acc_after = evaluate_model(interp_after)['accuracy']
        Z_after = probs.reshape(xx.shape)

        plot_confidence_boundary(ax_after, xx, yy, Z_after)
        for point, label in zip(XOR_DATA.numpy(), labels):
            ax_after.scatter(point[0], point[1], c=colors[label], s=150,
                             edgecolors='black', linewidths=2, zorder=5)

        ax_before.set_xlim(x_min, x_max)
        ax_before.set_ylim(y_min, y_max)
        ax_after.set_xlim(x_min, x_max)
        ax_after.set_ylim(y_min, y_max)
        ax_before.set_aspect('equal')
        ax_after.set_aspect('equal')

        if idx == 0:
            ax_before.set_ylabel('x₂ (before)', fontsize=10)
            ax_after.set_ylabel('x₂ (after)', fontsize=10)
        ax_after.set_xlabel('x₁', fontsize=10)

        ax_before.set_title(
            f't={t:.2f} | acc(before/after)={acc_before:.0f}%/{acc_after:.0f}%',
            fontsize=10
        )

    plt.suptitle('')
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_loss_curve_along_interpolation(before_results, after_results, seed_a, seed_b, output_path):
    """Plot loss vs interpolation t for before and after alignment."""
    ts = before_results['t']
    plt.figure(figsize=(6, 4))
    plt.plot(ts, before_results['loss'], label='Before alignment', color='#1f77b4', linewidth=2)
    plt.plot(ts, after_results['loss'], label='After alignment', color='#2ca02c', linewidth=2)
    plt.xlabel('Interpolation t', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title(f'Loss Along Interpolation (seed{seed_a} → seed{seed_b})', fontsize=11, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def compute_boundary_grid(model, grid_resolution=200):
    """Compute decision boundary grid for a binary XOR model."""
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution)
    )
    grid_points = torch.tensor(
        np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32
    )
    model.eval()
    with torch.no_grad():
        outputs = model(grid_points)
        probs = torch.sigmoid(outputs).squeeze(1).numpy()
    Z = probs.reshape(xx.shape)
    return xx, yy, Z


def save_boundary_plot(xx, yy, Z, output_path):
    """Save a single decision boundary plot to disk."""
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plot_confidence_boundary(ax, xx, yy, Z)

    labels = XOR_LABELS.long().squeeze(1).numpy()
    colors = ['#d62728', '#2ca02c']
    for point, label in zip(XOR_DATA.numpy(), labels):
        ax.scatter(point[0], point[1], c=colors[label], s=150,
                   edgecolors='black', linewidths=2, zorder=5)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlabel('x₁', fontsize=10)
    ax.set_ylabel('x₂', fontsize=10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def create_training_animation(frames, xx, yy, seed, output_path):
    """Create HTML animation of decision boundary evolution during training."""
    xor_x = XOR_DATA[:, 0].numpy()
    xor_y = XOR_DATA[:, 1].numpy()
    xor_labels = XOR_LABELS.long().squeeze(1).numpy()
    point_colors = ['#d62728' if l == 0 else '#2ca02c' for l in xor_labels]

    first = frames[0]
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=first['Z'],
        zmin=0.0,
        zmax=1.0,
        showscale=True,
        colorbar=dict(title='P(y=1)'),
        colorscale='RdYlGn',
        contours=dict(start=0.0, end=1.0, size=0.025, coloring='heatmap', showlines=False),
        hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<br>P(y=1)=%{z:.3f}<extra></extra>',
    ))
    fig.add_trace(go.Contour(
        x=xx[0],
        y=yy[:, 0],
        z=first['Z'],
        zmin=0.0,
        zmax=1.0,
        showscale=False,
        contours=dict(start=0.5, end=0.5, size=1, coloring='lines', showlabels=False),
        line=dict(width=3, color='black'),
        hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=xor_x,
        y=xor_y,
        mode='markers+text',
        marker=dict(size=20, color=point_colors, line=dict(width=2, color='black')),
        text=[f'({int(x)},{int(y)})={l}' for x, y, l in zip(xor_x, xor_y, xor_labels)],
        textposition='top right',
        textfont=dict(size=12),
        hoverinfo='skip'
    ))

    frame_objs = []
    for frame in frames:
        epoch = frame['epoch']
        acc = frame['accuracy']
        frame_objs.append(go.Frame(
            data=[
                go.Contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=frame['Z'],
                    zmin=0.0,
                    zmax=1.0,
                    showscale=False,
                    colorscale='RdYlGn',
                    contours=dict(start=0.0, end=1.0, size=0.025, coloring='heatmap', showlines=False),
                    hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<br>P(y=1)=%{z:.3f}<extra></extra>',
                ),
                go.Contour(
                    x=xx[0],
                    y=yy[:, 0],
                    z=frame['Z'],
                    zmin=0.0,
                    zmax=1.0,
                    showscale=False,
                    contours=dict(start=0.5, end=0.5, size=1, coloring='lines', showlabels=False),
                    line=dict(width=3, color='black'),
                    hoverinfo='skip'
                ),
                go.Scatter(
                    x=xor_x,
                    y=xor_y,
                    mode='markers+text',
                    marker=dict(size=20, color=point_colors, line=dict(width=2, color='black')),
                    text=[f'({int(x)},{int(y)})={l}' for x, y, l in zip(xor_x, xor_y, xor_labels)],
                    textposition='top right',
                    textfont=dict(size=12),
                    hoverinfo='skip'
                )
            ],
            name=f'epoch={epoch}',
            layout=go.Layout(
                title=dict(
                    text=f'Training boundary: seed{seed} | epoch {epoch} | acc {acc:.0f}%',
                    font=dict(size=16)
                )
            )
        ))

    fig.frames = frame_objs
    steps = [
        dict(
            args=[[f'epoch={f["epoch"]}'],
                  dict(mode='immediate', frame=dict(duration=80, redraw=True), transition=dict(duration=30))],
            label=str(f['epoch']),
            method='animate'
        )
        for f in frames
    ]

    fig.update_layout(
        title=dict(
            text=f'Training boundary: seed{seed} | epoch {first["epoch"]} | acc {first["accuracy"]:.0f}%',
            font=dict(size=16),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(title='x₁', range=[-0.5, 1.5], scaleanchor='y', scaleratio=1, dtick=0.5),
        yaxis=dict(title='x₂', range=[-0.5, 1.5], dtick=0.5),
        sliders=[dict(
            active=0,
            currentvalue=dict(prefix='epoch = ', visible=True),
            pad=dict(b=10, t=50),
            steps=steps
        )],
        updatemenus=[dict(
            type='buttons',
            showactive=False,
            y=1.15,
            x=0.0,
            xanchor='left',
            buttons=[
                dict(label='▶ Play', method='animate',
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      fromcurrent=True, transition=dict(duration=30))]),
                dict(label='⏸ Pause', method='animate',
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode='immediate')]),
            ]
        )],
        width=700,
        height=700,
        showlegend=False,
        margin=dict(t=120, b=80)
    )

    fig.write_html(output_path, include_plotlyjs='cdn')


# =============================================================================
# Main Experiment
# =============================================================================
def run_experiment(num_networks, output_dir, hidden_size=4, verbose=True,
                   track_seed=None, track_interval=50, track_grid=200,
                   plot_interpolation_before=False, plot_interpolation_after=False,
                   seeds=None):
    """Run the full XOR experiment.

    Args:
        num_networks: Number of networks to train
        output_dir: Directory for outputs
        hidden_size: Number of hidden neurons (default: 4)
        verbose: Print progress

    Returns:
        Dictionary with all results
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoints_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # =========================================================================
    # Step 1: Train networks
    # =========================================================================
    if seeds is None:
        seeds = list(range(num_networks))
    print("=" * 70)
    print(f"Step 1: Training {len(seeds)} networks on XOR ({hidden_size} hidden neurons)")
    print("=" * 70)

    models = {}
    model_info = {}  # Store loss and accuracy info

    for seed in seeds:
        track_config = None
        if track_seed is not None and seed == track_seed:
            frames_dir = os.path.join(output_dir, 'training_frames', f'seed{seed}')
            os.makedirs(frames_dir, exist_ok=True)
            track_config = {
                'interval': track_interval,
                'grid_resolution': track_grid,
                'frames_dir': frames_dir,
                'frames': [],
                'last_epoch': None,
            }
        if verbose:
            print(f"Training network with seed {seed}...")
        model, final_loss, final_acc, stop_reason, epochs_trained = train_xor_network(
            seed,
            hidden_size=hidden_size,
            verbose=verbose,
            track_config=track_config,
        )
        models[seed] = model
        model_info[seed] = {
            'loss': final_loss,
            'accuracy': final_acc,
            'stop_reason': stop_reason,
            'epochs_trained': epochs_trained,
        }
        # Save checkpoint
        torch.save({
            'model_state': model.state_dict(),
            'seed': seed,
            'hidden_size': hidden_size,
            'final_loss': final_loss,
            'final_accuracy': final_acc,
            'stop_reason': stop_reason,
            'epochs_trained': epochs_trained,
        }, os.path.join(checkpoints_dir, f'seed{seed}.pt'))

        if track_config is not None and track_config['frames']:
            animations_dir = os.path.join(output_dir, 'animations')
            os.makedirs(animations_dir, exist_ok=True)
            xx, yy, _ = compute_boundary_grid(model, grid_resolution=track_grid)
            animation_path = os.path.join(animations_dir, f'training_seed{seed}.html')
            create_training_animation(track_config['frames'], xx, yy, seed, animation_path)

    print(f"\nTrained {len(models)} networks")

    # Show final stats for all models
    print("\nFinal model statistics:")
    for seed in models:
        info = model_info[seed]
        if info['accuracy'] >= 100.0:
            params_vec = vectorize_model_params(models[seed]).cpu().numpy().tolist()
            print(
                f"  Seed {seed}: accuracy={info['accuracy']:.1f}%, loss={info['loss']:.6f}, "
                f"stop={info['stop_reason']}, epochs={info['epochs_trained']}"
            )
            print(f"    params={params_vec}")

    # Filter to only perfect-accuracy models for evaluation
    selected_seeds = [s for s, info in model_info.items() if info['accuracy'] >= 100.0]
    if len(selected_seeds) < len(models):
        removed = [s for s in models if s not in selected_seeds]
        print(f"\nFiltering to 100% accuracy models: keeping {len(selected_seeds)} / {len(models)}")
        if removed:
            print(f"  Removed seeds: {removed}")
    if not selected_seeds:
        print("\nNo models reached 100% accuracy. Skipping evaluation and plots.")
        return {
            'config': {
                'num_networks': len(models),
                'seed_list': list(models.keys()),
                'hidden_size': hidden_size,
                'architecture': f'2-{hidden_size}-1 MLP',
                'dataset': 'XOR (4 points)',
                'selected_seeds': [],
            },
            'training': {'model_info': model_info},
            'trained_seeds': list(models.keys()),
            'pair_results': [],
            'summary': {
                'num_pairs': 0,
                'barriers_before': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'barriers_after': {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0},
                'num_lmc_before': 0,
                'num_lmc_after': 0,
                'lmc_threshold': 1.0,
            }
        }

    models = {s: models[s] for s in selected_seeds}
    model_info = {s: model_info[s] for s in selected_seeds}

    # =========================================================================
    # Step 2: Evaluate all pairs
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 2: Evaluating barriers for all pairs")
    print("=" * 70)

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    pair_results = []
    seeds = list(models.keys())

    for seed_a, seed_b in combinations(seeds, 2):
        model_a = models[seed_a]
        model_b = models[seed_b]

        if verbose:
            print(f"\nPair ({seed_a}, {seed_b}):")

        # L2 distance before alignment
        dist_before = compute_l2_distance(model_a.state_dict(), model_b.state_dict())
        if verbose:
            print(f"  L2 distance: {dist_before['l2_distance']:.4f}")

        # Barrier before alignment
        barrier_before = compute_barrier(model_a, model_b)
        if verbose:
            print(f"  Before alignment: min_acc={barrier_before['min_accuracy']:.1f}%, barrier={barrier_before['barrier']:.1f}%")

        # Align models using exhaustive search
        model_b_aligned, best_perm, perm_results = align_models_exhaustive(model_a, model_b, verbose=verbose)

        # Verify alignment preserved model_b's accuracy
        res_b_original = evaluate_model(model_b)
        res_aligned = evaluate_model(model_b_aligned)
        if abs(res_aligned['accuracy'] - res_b_original['accuracy']) > 0.1:
            print(f"  WARNING: Alignment changed accuracy from {res_b_original['accuracy']:.1f}% to {res_aligned['accuracy']:.1f}%!")

        # L2 distance after alignment
        dist_after = compute_l2_distance(model_a.state_dict(), model_b_aligned.state_dict())
        if verbose:
            print(f"  L2 distance (after): {dist_after['l2_distance']:.4f}")

        # Barrier after alignment
        barrier_after = compute_barrier(model_a, model_b_aligned)
        if verbose:
            print(f"  After alignment: min_acc={barrier_after['min_accuracy']:.1f}%, barrier={barrier_after['barrier']:.1f}%")

        # Determine if permutation was identity
        identity_perm = list(range(len(best_perm)))
        perm_type = 'identity' if best_perm == identity_perm else 'swap'
        if verbose:
            print(f"  Best permutation: {best_perm} ({perm_type})")

        pair_results.append({
            'seed_a': seed_a,
            'seed_b': seed_b,
            'distance_before': dist_before,
            'distance_after': dist_after,
            'barrier_before': barrier_before,
            'barrier_after': barrier_after,
            'best_permutation': best_perm,
            'permutation_type': perm_type,
            'all_permutations': perm_results,
        })

        # Loss curve along interpolation (before/after)
        if plot_interpolation_before or plot_interpolation_after:
            loss_plot_path = os.path.join(plots_dir, f'loss_curve_{seed_a}_{seed_b}.png')
            plot_loss_curve_along_interpolation(barrier_before, barrier_after, seed_a, seed_b, loss_plot_path)

    # =========================================================================
    # Step 3: Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    barriers_before = [r['barrier_before']['barrier'] for r in pair_results]
    barriers_after = [r['barrier_after']['barrier'] for r in pair_results]
    perm_types = [r['permutation_type'] for r in pair_results]

    print(f"\nNumber of pairs: {len(pair_results)}")
    print(f"Permutation types: {sum(1 for p in perm_types if p == 'identity')} identity, {sum(1 for p in perm_types if p == 'swap')} swap")

    print(f"\nBarriers BEFORE alignment:")
    print(f"  Mean: {np.mean(barriers_before):.2f}%")
    print(f"  Std:  {np.std(barriers_before):.2f}%")
    print(f"  Min:  {np.min(barriers_before):.2f}%")
    print(f"  Max:  {np.max(barriers_before):.2f}%")

    print(f"\nBarriers AFTER alignment:")
    print(f"  Mean: {np.mean(barriers_after):.2f}%")
    print(f"  Std:  {np.std(barriers_after):.2f}%")
    print(f"  Min:  {np.min(barriers_after):.2f}%")
    print(f"  Max:  {np.max(barriers_after):.2f}%")

    # Check if one-basin hypothesis holds
    lmc_threshold = 1.0  # Consider LMC if barrier < 1%
    num_lmc_before = sum(1 for b in barriers_before if b < lmc_threshold)
    num_lmc_after = sum(1 for b in barriers_after if b < lmc_threshold)

    print(f"\nLMC pairs (barrier < {lmc_threshold}%):")
    print(f"  Before alignment: {num_lmc_before}/{len(pair_results)}")
    print(f"  After alignment:  {num_lmc_after}/{len(pair_results)}")

    if num_lmc_after == len(pair_results):
        print("\n✓ ONE-BASIN HYPOTHESIS HOLDS: All pairs are LMC after alignment!")
    elif num_lmc_after > num_lmc_before:
        print(f"\n~ PARTIAL SUPPORT: Alignment improved LMC connectivity ({num_lmc_before} -> {num_lmc_after})")
    else:
        print("\n✗ ONE-BASIN HYPOTHESIS REJECTED: Not all pairs are LMC after alignment")

    # =========================================================================
    # Save results
    # =========================================================================
    results = {
        'config': {
            'num_networks': len(seeds),
            'seed_list': seeds,
            'hidden_size': hidden_size,
            'architecture': f'2-{hidden_size}-1 MLP',
            'dataset': 'XOR (4 points)',
            'selected_seeds': seeds,
        },
        'training': {
            'model_info': model_info,
        },
        'trained_seeds': seeds,
        'pair_results': pair_results,
        'summary': {
            'num_pairs': len(pair_results),
            'barriers_before': {
                'mean': float(np.mean(barriers_before)),
                'std': float(np.std(barriers_before)),
                'min': float(np.min(barriers_before)),
                'max': float(np.max(barriers_before)),
            },
            'barriers_after': {
                'mean': float(np.mean(barriers_after)),
                'std': float(np.std(barriers_after)),
                'min': float(np.min(barriers_after)),
                'max': float(np.max(barriers_after)),
            },
            'num_lmc_before': num_lmc_before,
            'num_lmc_after': num_lmc_after,
            'lmc_threshold': lmc_threshold,
        }
    }

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # =========================================================================
    # Step 4: Visualizations
    # =========================================================================
    print("\n" + "=" * 70)
    print("Step 4: Generating visualizations")
    print("=" * 70)

    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Plot decision boundaries for all networks
    plot_decision_boundaries(
        models,
        os.path.join(plots_dir, 'decision_boundaries.png')
    )

    # Plot interpolation for every pair (before/after alignment as configured)
    if plot_interpolation_before or plot_interpolation_after:
        for pr in pair_results:
            seed_a = pr['seed_a']
            seed_b = pr['seed_b']
            model_a = models[seed_a]
            model_b = models[seed_b]

            best_perm = pr['best_permutation']
            aligned_state = apply_permutation_to_state(model_b.state_dict(), best_perm)
            model_b_aligned = SimpleMLP(hidden_size=hidden_size)
            model_b_aligned.load_state_dict(aligned_state)

            plot_interpolation_boundaries_stacked(
                model_a, model_b, model_b_aligned, seed_a, seed_b,
                os.path.join(plots_dir, f'interpolation_{seed_a}_{seed_b}_before_after.png')
            )

    return results


def main():
    parser = argparse.ArgumentParser(description='Pure XOR Experiment')
    parser.add_argument('--num-networks', type=int, default=5,
                        help='Number of networks to train (default: 5)')
    parser.add_argument('--hidden-neurons', type=int, default=4,
                        help='Number of hidden neurons (default: 4)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated list of seeds to train (overrides --num-networks)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: results/xor or results/xor_Nh)')
    parser.add_argument('--track-seed', type=int, default=None,
                        help='Seed index to track decision boundary during training')
    parser.add_argument('--track-interval', type=int, default=50,
                        help='Epoch interval for boundary snapshots (default: 50)')
    parser.add_argument('--track-grid', type=int, default=200,
                        help='Grid resolution for boundary snapshots (default: 200)')
    parser.add_argument('--plot-interpolation', action='store_true',
                        help='Generate interpolation plots for every pair (both before/after)')
    parser.add_argument('--plot-interpolation-before', action='store_true',
                        help='Generate interpolation plots before alignment')
    parser.add_argument('--plot-interpolation-after', action='store_true',
                        help='Generate interpolation plots after alignment')
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose output')
    args = parser.parse_args()

    # Default output directory based on hidden neurons
    if args.output is None:
        output_dir = f'results/xor_{args.hidden_neurons}h'
    else:
        output_dir = args.output

    seed_list = None
    if args.seeds:
        seed_list = [int(s) for s in args.seeds.split(',') if s.strip() != '']
        if not seed_list:
            raise ValueError("--seeds provided but no valid entries found")

    if args.track_seed is not None:
        if seed_list is not None:
            if args.track_seed not in seed_list:
                raise ValueError("--track-seed must be included in --seeds list")
        elif args.track_seed >= args.num_networks:
            raise ValueError("--track-seed must be less than --num-networks")

    plot_before = args.plot_interpolation_before
    plot_after = args.plot_interpolation_after
    if args.plot_interpolation:
        plot_before = True
        plot_after = True

    run_experiment(
        args.num_networks,
        output_dir,
        hidden_size=args.hidden_neurons,
        verbose=args.verbose,
        track_seed=args.track_seed,
        track_interval=args.track_interval,
        track_grid=args.track_grid,
        plot_interpolation_before=plot_before,
        plot_interpolation_after=plot_after,
        seeds=seed_list,
    )


if __name__ == '__main__':
    main()
