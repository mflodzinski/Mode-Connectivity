"""
Run PyTorch-AutoNEB on XOR checkpoints and export a curve.npz path.
"""

import argparse
import json
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from networkx import MultiGraph


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "external", "PyTorch-AutoNEB"))

from torch_autoneb import auto_neb  # noqa: E402
from torch_autoneb.config import AutoNEBConfig, NEBConfig, OptimConfig  # noqa: E402
from torch_autoneb.fill import equal, highest  # noqa: E402
from torch_autoneb.models import ModelWrapper  # noqa: E402


XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS_FLOAT = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
XOR_LABELS_LONG = XOR_LABELS_FLOAT.squeeze(1).long()


class XorNet(nn.Module):
    """2-H-1 XOR network that matches saved checkpoints."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h)


class XorAutoNEBModel(nn.Module):
    """AutoNEB-compatible model wrapper with train/test metrics keys."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = XorNet(hidden_size)

    def _log_probs(self) -> torch.Tensor:
        logits = self.net(XOR_DATA)
        two_class_logits = torch.cat([-logits, logits], dim=1)
        return F.log_softmax(two_class_logits, dim=1)

    def forward(self, **kwargs) -> torch.Tensor:
        log_probs = self._log_probs()
        return F.nll_loss(log_probs, XOR_LABELS_LONG)

    def analyse(self) -> Dict[str, float]:
        with torch.no_grad():
            log_probs = self._log_probs()
            loss = F.nll_loss(log_probs, XOR_LABELS_LONG).item()
            pred = torch.argmax(log_probs, dim=1)
            err = 1.0 - (pred == XOR_LABELS_LONG).float().mean().item()
        return {
            "train_loss": float(loss),
            "train_error": float(err),
            "test_loss": float(loss),
            "test_error": float(err),
        }


def load_checkpoint_state(path: str, hidden_size: int) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model_state"]
    ckpt_hidden_size = state["fc1.weight"].shape[0]
    ckpt_output_size = state["fc2.weight"].shape[0]
    if ckpt_hidden_size != hidden_size:
        raise ValueError(
            f"{path}: hidden size mismatch (checkpoint={ckpt_hidden_size}, expected={hidden_size})"
        )
    if ckpt_output_size != 1:
        raise ValueError(f"{path}: expected output size 1, got {ckpt_output_size}")
    return state


def state_to_vector(state: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            state["fc1.weight"].reshape(-1),
            state["fc1.bias"].reshape(-1),
            state["fc2.weight"].reshape(-1),
            state["fc2.bias"].reshape(-1),
        ]
    ).float()


def eval_param_vector(param_vector: torch.Tensor, hidden_size: int) -> Tuple[float, float]:
    idx = 0
    fc1_w = param_vector[idx : idx + hidden_size * 2].view(hidden_size, 2)
    idx += hidden_size * 2
    fc1_b = param_vector[idx : idx + hidden_size]
    idx += hidden_size
    fc2_w = param_vector[idx : idx + hidden_size].view(1, hidden_size)
    idx += hidden_size
    fc2_b = param_vector[idx : idx + 1]

    hidden = torch.relu(XOR_DATA @ fc1_w.t() + fc1_b)
    logits = hidden @ fc2_w.t() + fc2_b
    loss = F.binary_cross_entropy_with_logits(logits, XOR_LABELS_FLOAT).item()
    pred = (torch.sigmoid(logits) >= 0.5).long()
    acc = (pred == XOR_LABELS_FLOAT.long()).float().mean().item() * 100.0
    return float(loss), float(acc)


def sample_polyline(path_coords: torch.Tensor, num_points: int) -> np.ndarray:
    pivots = path_coords.detach().cpu().numpy()
    if pivots.shape[0] == 1:
        return np.repeat(pivots, repeats=num_points, axis=0)

    diffs = pivots[1:] - pivots[:-1]
    seg_lens = np.linalg.norm(diffs, axis=1)
    total_len = float(np.sum(seg_lens))
    if total_len <= 0:
        return np.repeat(pivots[:1], repeats=num_points, axis=0)

    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    s_targets = np.linspace(0.0, total_len, num_points)
    out = np.zeros((num_points, pivots.shape[1]), dtype=np.float64)
    seg = 0
    for i, s in enumerate(s_targets):
        while seg < len(seg_lens) - 1 and s > cum[seg + 1]:
            seg += 1
        denom = seg_lens[seg] if seg_lens[seg] > 0 else 1.0
        alpha = (s - cum[seg]) / denom
        alpha = min(max(alpha, 0.0), 1.0)
        out[i] = (1.0 - alpha) * pivots[seg] + alpha * pivots[seg + 1]
    return out


def path_stats(values: np.ndarray, dl: np.ndarray) -> Tuple[float, float, float, float]:
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    avg_val = float(np.mean(values))
    if len(values) > 1 and np.sum(dl[1:]) > 0:
        int_val = float(np.sum(0.5 * (values[:-1] + values[1:]) * dl[1:]) / np.sum(dl[1:]))
    else:
        int_val = avg_val
    return min_val, max_val, avg_val, int_val


def build_curve_npz_payload(ts: np.ndarray, losses: np.ndarray, accuracies: np.ndarray, vectors: np.ndarray):
    tr_loss = losses.astype(np.float64)
    te_loss = tr_loss.copy()
    tr_acc = accuracies.astype(np.float64)
    te_acc = tr_acc.copy()
    tr_err = 100.0 - tr_acc
    te_err = 100.0 - te_acc
    tr_nll = tr_loss.copy()
    te_nll = te_loss.copy()

    dl = np.zeros(len(ts), dtype=np.float64)
    if len(ts) > 1:
        diffs = vectors[1:] - vectors[:-1]
        dl[1:] = np.linalg.norm(diffs, axis=1)
    l2_norm = np.linalg.norm(vectors, axis=1).astype(np.float64)

    tr_loss_min, tr_loss_max, tr_loss_avg, tr_loss_int = path_stats(tr_loss, dl)
    tr_nll_min, tr_nll_max, tr_nll_avg, tr_nll_int = path_stats(tr_nll, dl)
    tr_err_min, tr_err_max, tr_err_avg, tr_err_int = path_stats(tr_err, dl)
    te_loss_min, te_loss_max, te_loss_avg, te_loss_int = path_stats(te_loss, dl)
    te_nll_min, te_nll_max, te_nll_avg, te_nll_int = path_stats(te_nll, dl)
    te_err_min, te_err_max, te_err_avg, te_err_int = path_stats(te_err, dl)

    return {
        "ts": ts,
        "param_vectors": vectors,
        "dl": dl,
        "l2_norm": l2_norm,
        "tr_loss": tr_loss,
        "tr_loss_min": tr_loss_min,
        "tr_loss_max": tr_loss_max,
        "tr_loss_avg": tr_loss_avg,
        "tr_loss_int": tr_loss_int,
        "tr_nll": tr_nll,
        "tr_nll_min": tr_nll_min,
        "tr_nll_max": tr_nll_max,
        "tr_nll_avg": tr_nll_avg,
        "tr_nll_int": tr_nll_int,
        "tr_acc": tr_acc,
        "tr_err": tr_err,
        "tr_err_min": tr_err_min,
        "tr_err_max": tr_err_max,
        "tr_err_avg": tr_err_avg,
        "tr_err_int": tr_err_int,
        "te_loss": te_loss,
        "te_loss_min": te_loss_min,
        "te_loss_max": te_loss_max,
        "te_loss_avg": te_loss_avg,
        "te_loss_int": te_loss_int,
        "te_nll": te_nll,
        "te_nll_min": te_nll_min,
        "te_nll_max": te_nll_max,
        "te_nll_avg": te_nll_avg,
        "te_nll_int": te_nll_int,
        "te_acc": te_acc,
        "te_err": te_err,
        "te_err_min": te_err_min,
        "te_err_max": te_err_max,
        "te_err_avg": te_err_avg,
        "te_err_int": te_err_int,
    }


def main():
    parser = argparse.ArgumentParser(description="Run AutoNEB for XOR checkpoint pair")
    parser.add_argument("--checkpoints-dir", type=str, required=True, help="Directory with seed{N}.pt files")
    parser.add_argument("--seed-a", type=int, required=True, help="First seed")
    parser.add_argument("--seed-b", type=int, required=True, help="Second seed")
    parser.add_argument("--hidden-neurons", type=int, default=3, help="Hidden size (default: 3)")
    parser.add_argument("--curve-eval-points", type=int, default=61, help="Export points in curve.npz")
    parser.add_argument("--output-root", type=str, default="results/xor_autoneb", help="Output root dir")
    parser.add_argument("--steps1", type=int, default=120, help="Cycle 1 optimizer steps")
    parser.add_argument("--steps2", type=int, default=120, help="Cycle 2 optimizer steps")
    parser.add_argument("--steps3", type=int, default=120, help="Cycle 3 optimizer steps")
    parser.add_argument("--lr1", type=float, default=0.05, help="Cycle 1-2 learning rate")
    parser.add_argument("--lr2", type=float, default=0.02, help="Cycle 3 learning rate")
    parser.add_argument("--insert-count-initial", type=int, default=2, help="Insert count for cycle 1 (equal)")
    parser.add_argument("--insert-count-refine", type=int, default=3, help="Insert count for cycles 2-3 (highest)")
    args = parser.parse_args()

    checkpoints_dir = args.checkpoints_dir
    if not os.path.isabs(checkpoints_dir):
        checkpoints_dir = os.path.join(project_root, checkpoints_dir)
    output_root = args.output_root
    if not os.path.isabs(output_root):
        output_root = os.path.join(project_root, output_root)

    ckpt_a = os.path.join(checkpoints_dir, f"seed{args.seed_a}.pt")
    ckpt_b = os.path.join(checkpoints_dir, f"seed{args.seed_b}.pt")
    if not os.path.exists(ckpt_a):
        raise FileNotFoundError(ckpt_a)
    if not os.path.exists(ckpt_b):
        raise FileNotFoundError(ckpt_b)

    base_model = XorAutoNEBModel(hidden_size=args.hidden_neurons)
    model = ModelWrapper(base_model)

    state_a = load_checkpoint_state(ckpt_a, args.hidden_neurons)
    base_model.net.load_state_dict(state_a)
    coords_a = model.get_coords().cpu()
    analysis_a = model.analyse()

    state_b = load_checkpoint_state(ckpt_b, args.hidden_neurons)
    base_model.net.load_state_dict(state_b)
    coords_b = model.get_coords().cpu()
    analysis_b = model.analyse()

    graph = MultiGraph()
    graph.add_node(args.seed_a, coords=coords_a, **analysis_a)
    graph.add_node(args.seed_b, coords=coords_b, **analysis_b)

    optim1 = OptimConfig(
        nsteps=args.steps1,
        algorithm_type=torch.optim.SGD,
        algorithm_args={"lr": args.lr1},
        scheduler_type=None,
        scheduler_args=None,
        eval_config=None,
    )
    optim2 = OptimConfig(
        nsteps=args.steps3,
        algorithm_type=torch.optim.SGD,
        algorithm_args={"lr": args.lr2},
        scheduler_type=None,
        scheduler_args=None,
        eval_config=None,
    )

    neb_configs = [
        NEBConfig(float("inf"), 0.0, equal, {"count": args.insert_count_initial}, 1, optim1),
        NEBConfig(float("inf"), 0.0, highest, {"count": args.insert_count_refine, "key": "dense_train_loss"}, 1, OptimConfig(args.steps2, torch.optim.SGD, {"lr": args.lr1}, None, None, None)),
        NEBConfig(float("inf"), 0.0, highest, {"count": args.insert_count_refine, "key": "dense_train_loss"}, 1, optim2),
    ]
    auto_cfg = AutoNEBConfig(neb_configs)

    auto_neb(args.seed_a, args.seed_b, graph, model, auto_cfg)

    edge_data = graph[args.seed_a][args.seed_b]
    best_cycle = min(edge_data.keys(), key=lambda k: edge_data[k]["saddle_train_loss"])
    best = edge_data[best_cycle]
    pivots = best["path_coords"]

    vectors = sample_polyline(pivots, args.curve_eval_points)
    ts = np.linspace(0.0, 1.0, args.curve_eval_points, dtype=np.float64)
    losses = np.zeros(args.curve_eval_points, dtype=np.float64)
    accs = np.zeros(args.curve_eval_points, dtype=np.float64)
    for i in range(args.curve_eval_points):
        l, a = eval_param_vector(torch.tensor(vectors[i], dtype=torch.float32), args.hidden_neurons)
        losses[i] = l
        accs[i] = a

    barrier = float(100.0 - np.min(accs))
    endpoint_avg_loss = float((losses[0] + losses[-1]) / 2.0)
    loss_barrier = float(np.max(losses) - endpoint_avg_loss)

    out_dir = os.path.join(
        output_root,
        "evaluations",
        f"seed{args.seed_a}-seed{args.seed_b}",
        "autoneb",
    )
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, "curve.npz")
    np.savez(npz_path, **build_curve_npz_payload(ts, losses, accs, vectors))

    summary = {
        "seed_a": int(args.seed_a),
        "seed_b": int(args.seed_b),
        "checkpoint_a": ckpt_a,
        "checkpoint_b": ckpt_b,
        "best_cycle": int(best_cycle),
        "num_cycles": int(len(edge_data)),
        "path_pivots": int(pivots.shape[0]),
        "curve_eval_points": int(args.curve_eval_points),
        "min_accuracy": float(np.min(accs)),
        "barrier": barrier,
        "max_loss": float(np.max(losses)),
        "endpoint_avg_loss": endpoint_avg_loss,
        "loss_barrier": loss_barrier,
        "autoneb_saddle_train_loss": float(best["saddle_train_loss"]),
        "autoneb_saddle_train_error": float(best["saddle_train_error"]),
        "autoneb_saddle_test_loss": float(best["saddle_test_loss"]),
        "autoneb_saddle_test_error": float(best["saddle_test_error"]),
        "npz_path": npz_path,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("=" * 70)
    print(f"AutoNEB completed for pair ({args.seed_a}, {args.seed_b})")
    print(f"Best cycle: {best_cycle}/{len(edge_data)}")
    print(f"Pivots in best path: {pivots.shape[0]}")
    print(f"Barrier: {barrier:.2f}%")
    print(f"Loss barrier: {loss_barrier:.6f}")
    print(f"curve.npz: {npz_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

