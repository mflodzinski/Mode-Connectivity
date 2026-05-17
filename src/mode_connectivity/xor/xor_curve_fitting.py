"""
Unified XOR connectivity experiment in one file.

Pipeline:
1. Train (or load) N seed models.
2. Keep only endpoints with 100% accuracy and low loss.
3. Run exhaustive hidden permutation search (selection: error barrier -> loss barrier -> L2).
4. Fit Bezier and PolyChain low-loss paths (num_bends configurable, default 3).
5. Compare permutation-aligned linear barrier vs best low-loss-path barrier.
6. Repeat for every eligible seed pair.
7. Plot dependency for both error barrier and loss barrier:
   barrier_after_permute vs low_loss_path_barrier.
"""

import argparse
import json
import os
import sys
import shutil
import math
from collections import OrderedDict
from itertools import combinations, permutations

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from mode_connectivity.common.paths import PROJECT_ROOT
from mode_connectivity.evaluation.metrics import state_distance_summary

XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


class SimpleMLP(nn.Module):
    """Simple MLP for XOR: 2 inputs -> H hidden -> C outputs (logits)."""

    def __init__(self, hidden_size=4, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def xor_loss_and_accuracy_from_logits(logits, output_size):
    """Compute XOR loss and accuracy for either 1-logit or 2-logit heads."""
    if output_size == 1:
        loss = F.binary_cross_entropy_with_logits(logits, XOR_LABELS)
        pred = (torch.sigmoid(logits) >= 0.5).long()
    elif output_size == 2:
        labels = XOR_LABELS.squeeze(1).long()
        loss = F.cross_entropy(logits, labels)
        pred = torch.argmax(logits, dim=1, keepdim=True).long()
    else:
        raise ValueError(f"Unsupported output_size={output_size}; expected 1 or 2")

    accuracy = (pred == XOR_LABELS.long()).float().mean().item() * 100.0
    return loss, accuracy


def evaluate_model(model):
    """Evaluate model on XOR dataset."""
    model.eval()
    with torch.no_grad():
        outputs = model(XOR_DATA)
        output_size = outputs.shape[1]
        loss, accuracy = xor_loss_and_accuracy_from_logits(outputs, output_size)
    return {'loss': float(loss.item()), 'accuracy': accuracy}


def get_training_config(hidden_size):
    """Get XOR training defaults."""
    return {
        'max_epochs': 5000,
        'lr': 0.05,
        'init_min': -0.5,
        'init_max': 0.5,
    }


def set_seed(seed):
    """Set RNG seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_xor_network(
    seed,
    hidden_size=3,
    max_epochs=None,
    lr=None,
    loss_threshold=1e-4,
    patience=500,
    lr_patience=200,
    lr_factor=0.5,
    verbose=False,
):
    """Train a 2-H-1 XOR network from scratch."""
    set_seed(seed)
    cfg = get_training_config(hidden_size)
    if max_epochs is None:
        max_epochs = cfg['max_epochs']
    else:
        max_epochs = min(max_epochs, cfg['max_epochs'])
    if lr is None:
        lr = cfg['lr']

    model = SimpleMLP(hidden_size=hidden_size, output_size=1)
    with torch.no_grad():
        for p in model.parameters():
            nn.init.uniform_(p, a=cfg['init_min'], b=cfg['init_max'])

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
    current_loss = float('inf')

    for epoch in range(max_epochs):
        optimizer.zero_grad()
        logits = model(XOR_DATA)
        loss = F.binary_cross_entropy_with_logits(logits, XOR_LABELS)
        loss.backward()
        optimizer.step()

        current_loss = float(loss.item())
        scheduler.step(current_loss)

        if best_loss - current_loss > loss_threshold:
            best_loss = current_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    eval_res = evaluate_model(model)
    if verbose:
        print(
            f"  Seed {seed}: trained acc={eval_res['accuracy']:.1f}% "
            f"loss={eval_res['loss']:.6f}"
        )
    return model, eval_res


def summarize_barriers(barriers):
    """Return basic summary statistics for a list of barriers."""
    if not barriers:
        return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return {
        'mean': float(np.mean(barriers)),
        'std': float(np.std(barriers)),
        'min': float(np.min(barriers)),
        'max': float(np.max(barriers)),
    }


def state_to_vector(state):
    """Flatten XOR model state dict into a single vector with fixed ordering."""
    return torch.cat([
        state['fc1.weight'].reshape(-1),
        state['fc1.bias'].reshape(-1),
        state['fc2.weight'].reshape(-1),
        state['fc2.bias'].reshape(-1),
    ]).float()


def logits_from_param_vector(x, param_vector, hidden_size, output_size):
    """Forward pass of 2-H-C MLP directly from flattened parameters."""
    idx = 0

    fc1_weight_size = hidden_size * 2
    fc1_weight = param_vector[idx:idx + fc1_weight_size].view(hidden_size, 2)
    idx += fc1_weight_size

    fc1_bias = param_vector[idx:idx + hidden_size]
    idx += hidden_size

    fc2_weight_size = output_size * hidden_size
    fc2_weight = param_vector[idx:idx + fc2_weight_size].view(output_size, hidden_size)
    idx += fc2_weight_size
    fc2_bias = param_vector[idx:idx + output_size]

    hidden = torch.relu(x @ fc1_weight.t() + fc1_bias)
    return hidden @ fc2_weight.t() + fc2_bias


def evaluate_param_vector(param_vector, hidden_size, output_size):
    """Evaluate XOR model defined by flattened parameters."""
    with torch.no_grad():
        logits = logits_from_param_vector(XOR_DATA, param_vector, hidden_size, output_size)
        loss, accuracy = xor_loss_and_accuracy_from_logits(logits, output_size)
    return {'loss': float(loss.item()), 'accuracy': accuracy}


def xor_margins_from_logits(logits, output_size):
    """Compute per-sample classification margins (higher is better)."""
    if output_size == 1:
        y_sign = 2.0 * XOR_LABELS.squeeze(1) - 1.0  # 0->-1, 1->+1
        return y_sign * logits.squeeze(1)
    if output_size == 2:
        labels = XOR_LABELS.squeeze(1).long()
        true_logits = logits[torch.arange(logits.shape[0]), labels]
        other_labels = 1 - labels
        other_logits = logits[torch.arange(logits.shape[0]), other_labels]
        return true_logits - other_logits
    raise ValueError(f"Unsupported output_size={output_size}; expected 1 or 2")


def curve_weights_from_control_points(t, control_points, curve_type):
    """Interpolate parameters from control points for Bezier or PolyChain."""
    curve = curve_type.lower()
    num_bends = len(control_points)
    if num_bends < 2:
        raise ValueError("Need at least 2 control points")

    if curve == 'bezier':
        n = num_bends - 1
        w_t = 0.0
        for i, point in enumerate(control_points):
            coeff = math.comb(n, i) * ((1.0 - t) ** (n - i)) * (t ** i)
            w_t = w_t + coeff * point
        return w_t

    if curve == 'polychain':
        t_n = t * (num_bends - 1)
        w_t = 0.0
        for i, point in enumerate(control_points):
            coeff = torch.clamp(1.0 - torch.abs(t_n - float(i)), min=0.0, max=1.0)
            w_t = w_t + coeff * point
        return w_t

    raise ValueError(f"Unsupported curve type: {curve_type}")


def evaluate_curve_path(control_points, hidden_size, output_size, curve_type, num_points=61):
    """Evaluate barrier along a fitted curve."""
    ts = np.linspace(0, 1, num_points)
    results = {
        'curve_type': curve_type,
        'num_bends': int(len(control_points)),
        'num_points': int(num_points),
        't': ts.tolist(),
        'loss': [],
        'accuracy': [],
    }

    for t in ts:
        t_tensor = torch.tensor(float(t), dtype=torch.float32)
        w_t = curve_weights_from_control_points(t_tensor, control_points, curve_type)
        eval_res = evaluate_param_vector(w_t, hidden_size, output_size)
        results['loss'].append(eval_res['loss'])
        results['accuracy'].append(eval_res['accuracy'])

    results['min_accuracy'] = min(results['accuracy'])
    results['max_loss'] = max(results['loss'])
    results['barrier'] = 100.0 - results['min_accuracy']
    results['endpoint_avg_loss'] = (results['loss'][0] + results['loss'][-1]) / 2.0
    results['loss_barrier'] = results['max_loss'] - results['endpoint_avg_loss']
    return results


def compute_linear_path(model_a, model_b, num_points=61):
    """Compute linear interpolation metrics between two models."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    hidden_size = state_a['fc1.weight'].shape[0]
    output_size = state_a['fc2.weight'].shape[0]
    ts = np.linspace(0, 1, num_points)
    results = {'t': ts.tolist(), 'loss': [], 'accuracy': []}

    for t in ts:
        interp_state = {}
        for key in state_a:
            interp_state[key] = (1.0 - t) * state_a[key] + t * state_b[key]
        interp_model = SimpleMLP(hidden_size=hidden_size, output_size=output_size)
        interp_model.load_state_dict(interp_state)
        eval_res = evaluate_model(interp_model)
        results['loss'].append(eval_res['loss'])
        results['accuracy'].append(eval_res['accuracy'])

    results['min_accuracy'] = min(results['accuracy'])
    results['max_loss'] = max(results['loss'])
    results['barrier'] = 100.0 - results['min_accuracy']
    results['endpoint_avg_loss'] = (results['loss'][0] + results['loss'][-1]) / 2.0
    results['loss_barrier'] = results['max_loss'] - results['endpoint_avg_loss']
    return results


def compute_l2_distance(state_a, state_b):
    """Compute L2 distance between two parameter states."""
    return state_distance_summary(state_a, state_b)


def apply_permutation_to_state(state, perm):
    """Apply hidden-neuron permutation to XOR MLP state dict."""
    perm = list(perm)
    out = OrderedDict()
    out['fc1.weight'] = state['fc1.weight'][perm, :]
    out['fc1.bias'] = state['fc1.bias'][perm]
    out['fc2.weight'] = state['fc2.weight'][:, perm]
    out['fc2.bias'] = state['fc2.bias'].clone()
    return out


def align_models_exhaustive(model_a, model_b, num_points=61, verbose=False):
    """Find best permutation of model_b hidden neurons relative to model_a."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    hidden_size = state_b['fc1.weight'].shape[0]
    output_size = state_b['fc2.weight'].shape[0]
    all_perms = list(permutations(range(hidden_size)))

    best_perm = None
    best_distance = float('inf')
    best_barrier = float('inf')
    best_loss_barrier = float('inf')
    perm_results = []

    if verbose:
        print(f"    Exhaustive permutations: {len(all_perms)}")

    for perm in all_perms:
        perm_state = apply_permutation_to_state(state_b, perm)
        dist = compute_l2_distance(state_a, perm_state)['l2_distance']

        temp_model = SimpleMLP(hidden_size=hidden_size, output_size=output_size)
        temp_model.load_state_dict(perm_state)
        barrier_res = compute_linear_path(model_a, temp_model, num_points=num_points)
        barrier = barrier_res['barrier']
        loss_barrier = barrier_res['loss_barrier']

        perm_results.append({
            'perm': list(perm),
            'l2_distance': float(dist),
            'barrier': float(barrier),
            'loss_barrier': float(loss_barrier),
        })

        better = False
        if barrier < best_barrier:
            better = True
        elif barrier == best_barrier:
            if loss_barrier < best_loss_barrier:
                better = True
            elif loss_barrier == best_loss_barrier and dist < best_distance:
                better = True

        if better:
            best_perm = perm
            best_distance = dist
            best_barrier = barrier
            best_loss_barrier = loss_barrier

    aligned_state = apply_permutation_to_state(state_b, best_perm)
    aligned_model = SimpleMLP(hidden_size=hidden_size, output_size=output_size)
    aligned_model.load_state_dict(aligned_state)
    return aligned_model, list(best_perm), perm_results


def fit_curve_between_models(model_a, model_b, curve_type='bezier',
                             steps=1500, lr=0.05, num_t_samples=11,
                             eval_points=61, weight_decay=0.0, verbose=False,
                             objective='mean_loss', margin_temperature=0.05,
                             num_bends=3):
    """Fit a curve with fixed endpoints between two XOR models."""
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    hidden_size = state_a['fc1.weight'].shape[0]
    output_size = state_a['fc2.weight'].shape[0]

    w0 = state_to_vector(state_a)
    w_end = state_to_vector(state_b)

    if num_bends < 3:
        raise ValueError("num_bends must be >= 3")

    # Endpoints are fixed, internal points are trainable.
    control_points = [w0]
    internal_points = []
    for i in range(1, num_bends - 1):
        alpha = i / (num_bends - 1)
        init_point = (1.0 - alpha) * w0 + alpha * w_end
        param = nn.Parameter(init_point.clone())
        internal_points.append(param)
        control_points.append(param)
    control_points.append(w_end)

    optimizer = torch.optim.Adam(internal_points, lr=lr, weight_decay=weight_decay)

    t_samples = torch.linspace(0.0, 1.0, num_t_samples)
    best_objective = float('inf')
    best_internal_points = [p.detach().clone() for p in internal_points]
    objective_name = objective

    if objective_name == 'min_margin' and margin_temperature <= 0:
        raise ValueError('margin_temperature must be > 0 for min_margin objective')

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        sampled_losses = []
        sampled_margins = []
        for t in t_samples:
            w_t = curve_weights_from_control_points(t, control_points, curve_type)
            logits = logits_from_param_vector(XOR_DATA, w_t, hidden_size, output_size)
            loss, _ = xor_loss_and_accuracy_from_logits(logits, output_size)
            sampled_losses.append(loss)
            sampled_margins.append(xor_margins_from_logits(logits, output_size))

        losses_tensor = torch.stack(sampled_losses)
        if objective_name == 'mean_loss':
            objective_tensor = losses_tensor.mean()
            log_metric = objective_tensor.item()
            log_label = 'mean_loss'
        elif objective_name == 'max_loss':
            objective_tensor = losses_tensor.max()
            log_metric = objective_tensor.item()
            log_label = 'max_loss'
        elif objective_name == 'min_margin':
            all_margins = torch.cat(sampled_margins, dim=0)
            # Smooth approximation of min(margins): softmin_tau(m)
            soft_min_margin = -margin_temperature * torch.logsumexp(
                -all_margins / margin_temperature, dim=0
            )
            objective_tensor = -soft_min_margin
            log_metric = soft_min_margin.item()
            log_label = 'soft_min_margin'
        else:
            raise ValueError(f"Unsupported curve objective: {objective_name}")

        objective_tensor.backward()
        optimizer.step()

        current_objective = objective_tensor.item()
        if current_objective < best_objective:
            best_objective = current_objective
            best_internal_points = [p.detach().clone() for p in internal_points]

        if verbose and (step % max(1, steps // 5) == 0 or step == 1 or step == steps):
            print(f"    [{curve_type}] step {step}/{steps}: {log_label}={log_metric:.6f}")

    fitted_control_points = [w0.detach().clone()] + best_internal_points + [w_end.detach().clone()]

    path_metrics = evaluate_curve_path(
        fitted_control_points,
        hidden_size=hidden_size,
        output_size=output_size,
        curve_type=curve_type,
        num_points=eval_points,
    )

    return {
        'fit': {
            'curve_type': curve_type,
            'steps': int(steps),
            'lr': float(lr),
            'num_t_samples': int(num_t_samples),
            'eval_points': int(eval_points),
            'weight_decay': float(weight_decay),
            'objective': objective_name,
            'margin_temperature': float(margin_temperature),
            'num_bends': int(num_bends),
            'num_trainable_points': int(max(0, num_bends - 2)),
            'best_objective': float(best_objective),
        },
        'path_metrics': path_metrics,
    }, fitted_control_points


def compute_path_vectors_linear(state_a, state_b, ts):
    """Compute flattened parameter vectors along linear interpolation path."""
    w0 = state_to_vector(state_a)
    w1 = state_to_vector(state_b)
    vectors = []
    for t in ts:
        t_float = float(t)
        vectors.append(((1.0 - t_float) * w0 + t_float * w1).cpu().numpy())
    return np.stack(vectors, axis=0)


def compute_path_vectors_curve(control_points, ts, curve_type):
    """Compute flattened parameter vectors along a fitted curve."""
    vectors = []
    for t in ts:
        t_tensor = torch.tensor(float(t), dtype=torch.float32)
        w_t = curve_weights_from_control_points(t_tensor, control_points, curve_type)
        vectors.append(w_t.detach().cpu().numpy())
    return np.stack(vectors, axis=0)


def path_stats(values, dl):
    """Compute start/end/min/max/avg/path-integral statistics."""
    vals = np.asarray(values, dtype=np.float64)
    min_val = float(np.min(vals))
    max_val = float(np.max(vals))
    avg_val = float(np.mean(vals))
    if len(vals) > 1 and np.sum(dl[1:]) > 0:
        int_val = float(np.sum(0.5 * (vals[:-1] + vals[1:]) * dl[1:]) / np.sum(dl[1:]))
    else:
        int_val = avg_val
    return min_val, max_val, avg_val, int_val


def build_curve_npz_payload(ts, losses, accuracies, path_vectors):
    """Build payload compatible with external eval_curve.py output."""
    ts = np.asarray(ts, dtype=np.float64)
    tr_loss = np.asarray(losses, dtype=np.float64)
    te_loss = np.asarray(losses, dtype=np.float64)
    tr_acc = np.asarray(accuracies, dtype=np.float64)
    te_acc = np.asarray(accuracies, dtype=np.float64)
    tr_err = 100.0 - tr_acc
    te_err = 100.0 - te_acc
    tr_nll = tr_loss.copy()
    te_nll = te_loss.copy()

    dl = np.zeros(len(ts), dtype=np.float64)
    if len(ts) > 1:
        diffs = path_vectors[1:] - path_vectors[:-1]
        dl[1:] = np.linalg.norm(diffs, axis=1)
    l2_norm = np.linalg.norm(path_vectors, axis=1).astype(np.float64)

    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = path_stats(tr_loss, dl)
    tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = path_stats(tr_nll, dl)
    tr_err_min, tr_err_max, tr_err_avg, tr_err_int = path_stats(tr_err, dl)
    te_loss_min, te_loss_max, te_loss_avg, te_loss_int = path_stats(te_loss, dl)
    te_nll_min, te_nll_max, te_nll_avg, te_nll_int = path_stats(te_nll, dl)
    te_err_min, te_err_max, te_err_avg, te_err_int = path_stats(te_err, dl)

    return {
        'ts': ts,
        'param_vectors': path_vectors,
        'dl': dl,
        'l2_norm': l2_norm,
        'tr_loss': tr_loss,
        'tr_loss_min': tr_loss_min,
        'tr_loss_max': tr_loss_max,
        'tr_loss_avg': tr_loss_avg,
        'tr_loss_int': tr_loss_int,
        'tr_nll': tr_nll,
        'tr_nll_min': tr_nll_min,
        'tr_nll_max': tr_nll_max,
        'tr_nll_avg': tr_nll_avg,
        'tr_nll_int': tr_nll_int,
        'tr_acc': tr_acc,
        'tr_err': tr_err,
        'tr_err_min': tr_err_min,
        'tr_err_max': tr_err_max,
        'tr_err_avg': tr_err_avg,
        'tr_err_int': tr_err_int,
        'te_loss': te_loss,
        'te_loss_min': te_loss_min,
        'te_loss_max': te_loss_max,
        'te_loss_avg': te_loss_avg,
        'te_loss_int': te_loss_int,
        'te_nll': te_nll,
        'te_nll_min': te_nll_min,
        'te_nll_max': te_nll_max,
        'te_nll_avg': te_nll_avg,
        'te_nll_int': te_nll_int,
        'te_acc': te_acc,
        'te_err': te_err,
        'te_err_min': te_err_min,
        'te_err_max': te_err_max,
        'te_err_avg': te_err_avg,
        'te_err_int': te_err_int,
    }


def save_curve_npz(npz_path, ts, losses, accuracies, path_vectors):
    """Save curve.npz in the same schema as external eval_curve.py output."""
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    payload = build_curve_npz_payload(ts, losses, accuracies, path_vectors)
    np.savez(npz_path, **payload)


def parse_pairs(pairs_arg, available_seeds):
    """Parse optional --pairs argument formatted as '2-4,2-5'."""
    if not pairs_arg:
        return list(combinations(sorted(available_seeds), 2))
    parsed = []
    for item in pairs_arg.split(','):
        item = item.strip()
        if not item:
            continue
        if '-' not in item:
            raise ValueError(f"Invalid pair '{item}', expected format 'a-b'")
        a_str, b_str = item.split('-', 1)
        a = int(a_str)
        b = int(b_str)
        if a == b:
            continue
        if a not in available_seeds or b not in available_seeds:
            raise ValueError(f"Pair {a}-{b} not present in loaded seeds")
        parsed.append((min(a, b), max(a, b)))
    return sorted(list(set(parsed)))


def parse_seed_list(seeds_arg, num_networks):
    """Parse seeds from --seeds or default to range(num_networks)."""
    if seeds_arg:
        seeds = [int(s) for s in seeds_arg.split(',') if s.strip()]
        if not seeds:
            raise ValueError('No valid seeds parsed from --seeds')
        return seeds
    return list(range(num_networks))


def save_checkpoint(path, seed, hidden_size, model, eval_res, source):
    """Save a standardized checkpoint payload."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': model.state_dict(),
        'seed': int(seed),
        'hidden_size': int(hidden_size),
        'final_loss': float(eval_res['loss']),
        'final_accuracy': float(eval_res['accuracy']),
        'source': source,
    }, path)


def load_or_train_models(seeds, hidden_size, checkpoints_dir, output_dir, train_max_epochs, train_lr, verbose):
    """Load checkpoints when available; otherwise train from scratch."""
    run_ckpt_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(run_ckpt_dir, exist_ok=True)

    models = {}
    model_info = {}
    for seed in seeds:
        source = 'trained'
        model = None
        eval_res = None
        loaded_ckpt_path = None

        if checkpoints_dir is not None:
            ckpt_path = os.path.join(checkpoints_dir, f'seed{seed}.pt')
            if os.path.exists(ckpt_path):
                try:
                    checkpoint = torch.load(ckpt_path, map_location='cpu')
                    state = checkpoint['model_state']
                    ckpt_hidden_size = state['fc1.weight'].shape[0]
                    ckpt_output_size = state['fc2.weight'].shape[0]
                    if ckpt_output_size != 1:
                        raise ValueError(
                            f"Expected 1 output logit, found {ckpt_output_size}"
                        )
                    if ckpt_hidden_size != hidden_size:
                        raise ValueError(
                            f"Hidden size mismatch (checkpoint={ckpt_hidden_size}, expected={hidden_size})"
                        )
                    model = SimpleMLP(hidden_size=hidden_size, output_size=1)
                    model.load_state_dict(state)
                    eval_res = evaluate_model(model)
                    source = 'checkpoint'
                    loaded_ckpt_path = ckpt_path
                except Exception as exc:
                    print(f"  Seed {seed}: checkpoint unusable ({exc}), retraining.")
            else:
                if verbose:
                    print(f"  Seed {seed}: checkpoint missing, retraining.")

        if model is None:
            model, eval_res = train_xor_network(
                seed,
                hidden_size=hidden_size,
                max_epochs=train_max_epochs,
                lr=train_lr,
                verbose=verbose,
            )

        ckpt_out_path = os.path.join(run_ckpt_dir, f'seed{seed}.pt')
        save_checkpoint(ckpt_out_path, seed, hidden_size, model, eval_res, source=source)

        models[seed] = model
        model_info[seed] = {
            'accuracy': float(eval_res['accuracy']),
            'loss': float(eval_res['loss']),
            'source': source,
            'loaded_checkpoint': loaded_ckpt_path,
            'saved_checkpoint': ckpt_out_path,
            'hidden_size': int(hidden_size),
            'output_size': 1,
        }
        print(
            f"  Seed {seed}: {source} "
            f"(accuracy={eval_res['accuracy']:.1f}%, loss={eval_res['loss']:.6f})"
        )

    return models, model_info, run_ckpt_dir


def filter_eligible_seeds(model_info, max_endpoint_loss):
    """Select seeds that satisfy point (2): 100% accuracy and low loss."""
    return sorted([
        seed for seed, info in model_info.items()
        if info['accuracy'] >= 99.999 and info['loss'] <= max_endpoint_loss
    ])


def choose_low_loss_path(bezier_metrics, poly_metrics):
    """Pick the 'low loss path' by loss barrier (then error barrier)."""
    candidates = {
        'bezier': bezier_metrics,
        'polychain': poly_metrics,
    }
    best_name = min(
        candidates.keys(),
        key=lambda name: (
            candidates[name]['loss_barrier'],
            candidates[name]['barrier'],
        ),
    )
    return best_name, candidates[best_name]


def _plot_dependency_scatter(x_vals, y_vals, labels, output_path, xlabel, ylabel, title):
    """Generic dependency scatter helper."""
    if len(x_vals) == 0:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(x_vals, y_vals, s=70, alpha=0.85, color='#1f77b4')
    for label, x, y in zip(labels, x_vals, y_vals):
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(5, 4), fontsize=8)

    upper = max(1.0, float(max(np.max(x_vals), np.max(y_vals)))) + 1.0
    ax.plot([0.0, upper], [0.0, upper], linestyle='--', linewidth=1.5, color='black', alpha=0.5)
    ax.set_xlim(0.0, upper)
    ax.set_ylim(0.0, upper)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if len(x_vals) >= 2 and np.std(x_vals) > 1e-12 and np.std(y_vals) > 1e-12:
        corr = float(np.corrcoef(x_vals, y_vals)[0, 1])
        ax.text(
            0.02, 0.98, f"Pearson r = {corr:.3f}",
            transform=ax.transAxes,
            va='top',
            ha='left',
            fontsize=10,
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75),
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def plot_barrier_dependency(pair_results, output_path):
    """Plot accuracy barrier dependency: permuted linear vs low-loss path."""
    if not pair_results:
        return
    x_vals = np.asarray([p['comparison']['permute_barrier'] for p in pair_results], dtype=np.float64)
    y_vals = np.asarray([p['comparison']['low_loss_path_barrier'] for p in pair_results], dtype=np.float64)
    labels = [f"{p['seed_a']}-{p['seed_b']}" for p in pair_results]
    _plot_dependency_scatter(
        x_vals,
        y_vals,
        labels,
        output_path,
        xlabel='Barrier after permutation (%)',
        ylabel='Low-loss path barrier (%)',
        title='Barrier Dependency: Permuted Linear vs Low-Loss Path',
    )


def plot_loss_barrier_dependency(pair_results, output_path):
    """Plot loss-barrier dependency: permuted linear vs low-loss path."""
    if not pair_results:
        return
    x_vals = np.asarray([p['comparison']['permute_loss_barrier'] for p in pair_results], dtype=np.float64)
    y_vals = np.asarray([p['comparison']['low_loss_path_loss_barrier'] for p in pair_results], dtype=np.float64)
    labels = [f"{p['seed_a']}-{p['seed_b']}" for p in pair_results]
    _plot_dependency_scatter(
        x_vals,
        y_vals,
        labels,
        output_path,
        xlabel='Loss barrier after permutation',
        ylabel='Low-loss path loss barrier',
        title='Loss-Barrier Dependency: Permuted Linear vs Low-Loss Path',
    )


def main():
    parser = argparse.ArgumentParser(description='Unified XOR connectivity pipeline')
    parser.add_argument('--checkpoints-dir', type=str, default=None,
                        help='Optional checkpoint dir to load from; missing/incompatible seeds are retrained')
    parser.add_argument('--num-networks', type=int, default=10,
                        help='Number of seeds when --seeds is omitted (default: 10)')
    parser.add_argument('--seeds', type=str, default=None,
                        help='Optional comma-separated seeds, e.g. 1,3,7,9')
    parser.add_argument('--pairs', type=str, default=None,
                        help='Optional comma-separated explicit pairs, e.g. 2-4,2-5')
    parser.add_argument('--hidden-neurons', type=int, default=3,
                        help='Hidden neurons for trained/loaded XOR models (default: 3)')
    parser.add_argument('--output', type=str, default='results/xor_unified',
                        help='Output root directory (default: results/xor)')
    parser.add_argument('--max-endpoint-loss', type=float, default=0.01,
                        help='Keep only seeds with loss <= this value and 100%% acc (default: 0.01)')
    parser.add_argument('--train-max-epochs', type=int, default=None,
                        help='Optional max epochs for retraining')
    parser.add_argument('--train-lr', type=float, default=None,
                        help='Optional learning rate for retraining')
    parser.add_argument('--curve-steps', type=int, default=1500,
                        help='Optimization steps for each fitted curve (default: 1500)')
    parser.add_argument('--curve-lr', type=float, default=0.05,
                        help='Shared learning rate for curve fitting (default: 0.05)')
    parser.add_argument('--bezier-lr', type=float, default=None,
                        help='Bezier learning rate (default: use --curve-lr)')
    parser.add_argument('--polychain-lr', type=float, default=None,
                        help='PolyChain learning rate (default: use --curve-lr)')
    parser.add_argument('--curve-t-samples', type=int, default=11,
                        help='t samples in fitting objective (default: 11)')
    parser.add_argument('--curve-eval-points', type=int, default=61,
                        help='Points for final path evaluation/npz export (default: 61)')
    parser.add_argument('--bezier-num-bends', type=int, default=3,
                        help='Number of bends for Bezier path (default: 3, trainable points = bends-2)')
    parser.add_argument('--polychain-num-bends', type=int, default=3,
                        help='Number of bends for PolyChain path (default: 3, trainable points = bends-2)')
    parser.add_argument('--curve-weight-decay', type=float, default=0.0,
                        help='Weight decay for curve fitting optimizer (default: 0.0)')
    parser.add_argument('--curve-objective', type=str, default='mean_loss',
                        choices=['mean_loss', 'max_loss', 'min_margin'],
                        help='Objective for fitting trainable bend point (default: mean_loss)')
    parser.add_argument('--margin-temperature', type=float, default=0.05,
                        help='Soft-min temperature for min_margin objective (default: 0.05)')
    parser.add_argument('--skip-plots', action='store_true', help='Skip summary plot generation')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    if args.hidden_neurons <= 0:
        raise ValueError('--hidden-neurons must be > 0')
    if args.num_networks <= 0:
        raise ValueError('--num-networks must be > 0')
    if args.max_endpoint_loss < 0:
        raise ValueError('--max-endpoint-loss must be >= 0')
    if args.curve_steps <= 0:
        raise ValueError('--curve-steps must be > 0')
    if args.curve_t_samples < 2:
        raise ValueError('--curve-t-samples must be >= 2')
    if args.curve_eval_points < 2:
        raise ValueError('--curve-eval-points must be >= 2')
    if args.bezier_num_bends < 3:
        raise ValueError('--bezier-num-bends must be >= 3')
    if args.polychain_num_bends < 3:
        raise ValueError('--polychain-num-bends must be >= 3')
    if args.curve_objective == 'min_margin' and args.margin_temperature <= 0:
        raise ValueError('--margin-temperature must be > 0 for min_margin objective')
    if args.curve_lr <= 0:
        raise ValueError('--curve-lr must be > 0')

    bezier_lr = args.bezier_lr if args.bezier_lr is not None else args.curve_lr
    polychain_lr = args.polychain_lr if args.polychain_lr is not None else args.curve_lr
    if bezier_lr <= 0:
        raise ValueError('--bezier-lr must be > 0')
    if polychain_lr <= 0:
        raise ValueError('--polychain-lr must be > 0')

    checkpoints_dir = args.checkpoints_dir
    if checkpoints_dir is not None and not os.path.isabs(checkpoints_dir):
        checkpoints_dir = os.path.join(str(PROJECT_ROOT), checkpoints_dir)
    output_dir = args.output
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(str(PROJECT_ROOT), output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    seeds = parse_seed_list(args.seeds, args.num_networks)
    print(f"Step 1/7: Build seed models for seeds: {seeds}")
    print("=" * 70)
    models, model_info, run_ckpt_dir = load_or_train_models(
        seeds=seeds,
        hidden_size=args.hidden_neurons,
        checkpoints_dir=checkpoints_dir,
        output_dir=output_dir,
        train_max_epochs=args.train_max_epochs,
        train_lr=args.train_lr,
        verbose=args.verbose,
    )

    print("\n" + "=" * 70)
    print(
        f"Step 2/7: Filter endpoints by accuracy=100% and loss<={args.max_endpoint_loss:.6f}"
    )
    print("=" * 70)
    selected_seeds = filter_eligible_seeds(model_info, args.max_endpoint_loss)
    rejected = [s for s in seeds if s not in selected_seeds]
    print(f"Eligible seeds: {selected_seeds}")
    if rejected:
        print(f"Rejected seeds: {rejected}")
    if len(selected_seeds) < 2:
        raise ValueError(
            f"Need at least 2 seeds after filtering; got {len(selected_seeds)}. "
            "Increase --num-networks/--seeds or relax --max-endpoint-loss."
        )

    pairs = parse_pairs(args.pairs, selected_seeds)
    if not pairs:
        raise ValueError('No valid pairs to evaluate')

    print("\n" + "=" * 70)
    print(f"Steps 3-6/7: permutation + curve fitting for {len(pairs)} pairs")
    print("=" * 70)

    evaluations_dir = os.path.join(output_dir, 'evaluations')
    os.makedirs(evaluations_dir, exist_ok=True)
    pair_results = []

    for seed_a, seed_b in pairs:
        model_a = models[seed_a]
        model_b = models[seed_b]
        pair_eval_dir = os.path.join(evaluations_dir, f'seed{seed_a}-seed{seed_b}')
        if os.path.isdir(pair_eval_dir):
            shutil.rmtree(pair_eval_dir)

        print(f"\nPair ({seed_a}, {seed_b})")
        linear_before = compute_linear_path(model_a, model_b, num_points=args.curve_eval_points)

        model_b_aligned, best_perm, perm_results = align_models_exhaustive(
            model_a,
            model_b,
            num_points=args.curve_eval_points,
            verbose=args.verbose,
        )
        linear_after = compute_linear_path(model_a, model_b_aligned, num_points=args.curve_eval_points)

        bezier, bezier_control_points = fit_curve_between_models(
            model_a, model_b,
            curve_type='bezier',
            steps=args.curve_steps,
            lr=bezier_lr,
            num_t_samples=args.curve_t_samples,
            eval_points=args.curve_eval_points,
            weight_decay=args.curve_weight_decay,
            objective=args.curve_objective,
            margin_temperature=args.margin_temperature,
            num_bends=args.bezier_num_bends,
            verbose=args.verbose,
        )

        polychain, polychain_control_points = fit_curve_between_models(
            model_a, model_b,
            curve_type='polychain',
            steps=args.curve_steps,
            lr=polychain_lr,
            num_t_samples=args.curve_t_samples,
            eval_points=args.curve_eval_points,
            weight_decay=args.curve_weight_decay,
            objective=args.curve_objective,
            margin_temperature=args.margin_temperature,
            num_bends=args.polychain_num_bends,
            verbose=args.verbose,
        )

        linear_before_ts = np.asarray(linear_before['t'], dtype=np.float64)
        linear_before_vectors = compute_path_vectors_linear(
            model_a.state_dict(),
            model_b.state_dict(),
            linear_before_ts,
        )
        linear_before_npz = os.path.join(pair_eval_dir, 'linear_before', 'curve.npz')
        save_curve_npz(
            linear_before_npz,
            linear_before_ts,
            linear_before['loss'],
            linear_before['accuracy'],
            linear_before_vectors,
        )

        linear_after_ts = np.asarray(linear_after['t'], dtype=np.float64)
        linear_after_vectors = compute_path_vectors_linear(
            model_a.state_dict(),
            model_b_aligned.state_dict(),
            linear_after_ts,
        )
        linear_after_npz = os.path.join(pair_eval_dir, 'linear_after', 'curve.npz')
        save_curve_npz(
            linear_after_npz,
            linear_after_ts,
            linear_after['loss'],
            linear_after['accuracy'],
            linear_after_vectors,
        )

        bezier_ts = np.asarray(bezier['path_metrics']['t'], dtype=np.float64)
        bezier_vectors = compute_path_vectors_curve(bezier_control_points, bezier_ts, 'bezier')
        bezier_npz = os.path.join(pair_eval_dir, 'bezier', 'curve.npz')
        save_curve_npz(
            bezier_npz,
            bezier_ts,
            bezier['path_metrics']['loss'],
            bezier['path_metrics']['accuracy'],
            bezier_vectors,
        )

        poly_ts = np.asarray(polychain['path_metrics']['t'], dtype=np.float64)
        poly_vectors = compute_path_vectors_curve(polychain_control_points, poly_ts, 'polychain')
        poly_npz = os.path.join(pair_eval_dir, 'polychain', 'curve.npz')
        save_curve_npz(
            poly_npz,
            poly_ts,
            polychain['path_metrics']['loss'],
            polychain['path_metrics']['accuracy'],
            poly_vectors,
        )

        low_loss_name, low_loss_metrics = choose_low_loss_path(
            bezier['path_metrics'],
            polychain['path_metrics'],
        )

        print(
            f"  barriers: linear_before={linear_before['barrier']:.1f}% "
            f"linear_after={linear_after['barrier']:.1f}% "
            f"bezier={bezier['path_metrics']['barrier']:.1f}% "
            f"polychain={polychain['path_metrics']['barrier']:.1f}%"
        )
        print(
            f"  compare: permuted={linear_after['barrier']:.1f}% "
            f"vs low_loss({low_loss_name})={low_loss_metrics['barrier']:.1f}%"
        )
        print(
            f"  loss barriers: permuted={linear_after['loss_barrier']:.6f} "
            f"vs low_loss({low_loss_name})={low_loss_metrics['loss_barrier']:.6f}"
        )

        pair_results.append({
            'seed_a': seed_a,
            'seed_b': seed_b,
            'linear_before': linear_before,
            'linear_after': linear_after,
            'best_permutation': best_perm,
            'all_permutations': perm_results,
            'bezier': bezier,
            'polychain': polychain,
            'low_loss_path': {
                'type': low_loss_name,
                'barrier': float(low_loss_metrics['barrier']),
                'loss_barrier': float(low_loss_metrics['loss_barrier']),
            },
            'comparison': {
                'permute_barrier': float(linear_after['barrier']),
                'low_loss_path_barrier': float(low_loss_metrics['barrier']),
                'low_loss_minus_permute': float(low_loss_metrics['barrier'] - linear_after['barrier']),
                'permute_loss_barrier': float(linear_after['loss_barrier']),
                'low_loss_path_loss_barrier': float(low_loss_metrics['loss_barrier']),
                'low_loss_minus_permute_loss': float(low_loss_metrics['loss_barrier'] - linear_after['loss_barrier']),
            },
            'npz_paths': {
                'linear_before': linear_before_npz,
                'linear_after': linear_after_npz,
                'bezier': bezier_npz,
                'polychain': poly_npz,
            }
        })

    linear_before_barriers = [p['linear_before']['barrier'] for p in pair_results]
    linear_after_barriers = [p['linear_after']['barrier'] for p in pair_results]
    bezier_barriers = [p['bezier']['path_metrics']['barrier'] for p in pair_results]
    poly_barriers = [p['polychain']['path_metrics']['barrier'] for p in pair_results]
    low_loss_barriers = [p['low_loss_path']['barrier'] for p in pair_results]
    deltas = [p['comparison']['low_loss_minus_permute'] for p in pair_results]
    linear_after_loss_barriers = [p['comparison']['permute_loss_barrier'] for p in pair_results]
    low_loss_loss_barriers = [p['comparison']['low_loss_path_loss_barrier'] for p in pair_results]
    delta_loss = [p['comparison']['low_loss_minus_permute_loss'] for p in pair_results]

    threshold = 1.0
    summary = {
        'num_pairs': len(pair_results),
        'barriers_linear_before': summarize_barriers(linear_before_barriers),
        'barriers_linear_after_permute': summarize_barriers(linear_after_barriers),
        'barriers_bezier': summarize_barriers(bezier_barriers),
        'barriers_polychain': summarize_barriers(poly_barriers),
        'barriers_low_loss_path': summarize_barriers(low_loss_barriers),
        'delta_low_loss_minus_permute': summarize_barriers(deltas),
        'loss_barriers_linear_after_permute': summarize_barriers(linear_after_loss_barriers),
        'loss_barriers_low_loss_path': summarize_barriers(low_loss_loss_barriers),
        'delta_loss_low_loss_minus_permute': summarize_barriers(delta_loss),
        'num_lmc_linear_before': sum(1 for b in linear_before_barriers if b < threshold),
        'num_lmc_linear_after_permute': sum(1 for b in linear_after_barriers if b < threshold),
        'num_lmc_bezier': sum(1 for b in bezier_barriers if b < threshold),
        'num_lmc_polychain': sum(1 for b in poly_barriers if b < threshold),
        'num_lmc_low_loss_path': sum(1 for b in low_loss_barriers if b < threshold),
        'num_pairs_low_loss_better_than_permute': int(sum(1 for d in deltas if d < 0.0)),
        'num_pairs_low_loss_better_than_permute_loss': int(sum(1 for d in delta_loss if d < 0.0)),
        'num_pairs_equal': int(sum(1 for d in deltas if abs(d) <= 1e-9)),
        'lmc_threshold': threshold,
    }

    if len(linear_after_barriers) >= 2 and np.std(linear_after_barriers) > 1e-12 and np.std(low_loss_barriers) > 1e-12:
        summary['pearson_permute_vs_low_loss'] = float(np.corrcoef(linear_after_barriers, low_loss_barriers)[0, 1])
    else:
        summary['pearson_permute_vs_low_loss'] = None
    if len(linear_after_loss_barriers) >= 2 and np.std(linear_after_loss_barriers) > 1e-12 and np.std(low_loss_loss_barriers) > 1e-12:
        summary['pearson_permute_vs_low_loss_loss_barrier'] = float(
            np.corrcoef(linear_after_loss_barriers, low_loss_loss_barriers)[0, 1]
        )
    else:
        summary['pearson_permute_vs_low_loss_loss_barrier'] = None

    plots_dir = os.path.join(output_dir, 'plots')
    barrier_dependency_plot = os.path.join(plots_dir, 'barrier_dependency_permute_vs_low_loss.png')
    loss_barrier_dependency_plot = os.path.join(plots_dir, 'loss_barrier_dependency_permute_vs_low_loss.png')
    if not args.skip_plots:
        plot_barrier_dependency(pair_results, barrier_dependency_plot)
        plot_loss_barrier_dependency(pair_results, loss_barrier_dependency_plot)

    results = {
        'config': {
            'seeds_requested': seeds,
            'seeds_used': selected_seeds,
            'pairs': [[a, b] for a, b in pairs],
            'hidden_size': int(args.hidden_neurons),
            'output_size': 1,
            'num_networks': int(args.num_networks),
            'max_endpoint_loss': float(args.max_endpoint_loss),
            'checkpoints_dir': checkpoints_dir,
            'run_checkpoints_dir': run_ckpt_dir,
            'output_dir': output_dir,
            'train': {
                'max_epochs': args.train_max_epochs,
                'lr': args.train_lr,
            },
            'curve_fit': {
                'steps': int(args.curve_steps),
                'curve_lr': float(args.curve_lr),
                'bezier_lr': float(bezier_lr),
                'polychain_lr': float(polychain_lr),
                'num_t_samples': int(args.curve_t_samples),
                'eval_points': int(args.curve_eval_points),
                'bezier_num_bends': int(args.bezier_num_bends),
                'polychain_num_bends': int(args.polychain_num_bends),
                'weight_decay': float(args.curve_weight_decay),
                'objective': args.curve_objective,
                'margin_temperature': float(args.margin_temperature),
            },
        },
        'model_info': model_info,
        'pair_results': pair_results,
        'summary': summary,
        'artifacts': {
            'evaluations_dir': evaluations_dir,
            'barrier_dependency_plot': None if args.skip_plots else barrier_dependency_plot,
            'loss_barrier_dependency_plot': None if args.skip_plots else loss_barrier_dependency_plot,
        },
    }

    results_path = os.path.join(output_dir, 'xor_unified_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("UNIFIED XOR SUMMARY")
    print("=" * 70)
    print(f"Eligible seeds: {selected_seeds}")
    print(f"Pairs evaluated: {summary['num_pairs']}")
    print(f"Curve LRs: bezier={bezier_lr:.6f}, polychain={polychain_lr:.6f}")
    print(f"Mean barrier linear_before:      {summary['barriers_linear_before']['mean']:.2f}%")
    print(f"Mean barrier linear_after_perm:  {summary['barriers_linear_after_permute']['mean']:.2f}%")
    print(f"Mean barrier Bezier:             {summary['barriers_bezier']['mean']:.2f}%")
    print(f"Mean barrier PolyChain:          {summary['barriers_polychain']['mean']:.2f}%")
    print(f"Mean barrier low-loss path:      {summary['barriers_low_loss_path']['mean']:.2f}%")
    print(f"Mean loss barrier linear_after:  {summary['loss_barriers_linear_after_permute']['mean']:.6f}")
    print(f"Mean loss barrier low-loss path: {summary['loss_barriers_low_loss_path']['mean']:.6f}")
    print(
        f"Pairs where low-loss path beats permutation: "
        f"{summary['num_pairs_low_loss_better_than_permute']}/{summary['num_pairs']}"
    )
    print(
        f"Pairs where low-loss path beats permutation (loss): "
        f"{summary['num_pairs_low_loss_better_than_permute_loss']}/{summary['num_pairs']}"
    )
    if summary['pearson_permute_vs_low_loss'] is not None:
        print(f"Pearson r (permute vs low-loss barriers): {summary['pearson_permute_vs_low_loss']:.3f}")
    if summary['pearson_permute_vs_low_loss_loss_barrier'] is not None:
        print(
            "Pearson r (permute vs low-loss loss barriers): "
            f"{summary['pearson_permute_vs_low_loss_loss_barrier']:.3f}"
        )
    print(f"Results: {results_path}")
    print(f"NPZ dir: {evaluations_dir}")
    if not args.skip_plots:
        print(f"Barrier dependency plot: {barrier_dependency_plot}")
        print(f"Loss-barrier dependency plot: {loss_barrier_dependency_plot}")


if __name__ == '__main__':
    main()
