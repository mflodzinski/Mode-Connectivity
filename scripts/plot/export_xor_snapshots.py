"""Export static XOR boundary snapshots from curve.npz."""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


XOR_DATA = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
XOR_LABELS = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)


def infer_hidden_size(num_params):
    """Infer hidden size H from 2-H-1 MLP parameter count (4H+1)."""
    if (num_params - 1) % 4 != 0:
        raise ValueError(f"Cannot infer hidden size from parameter count={num_params}")
    return (num_params - 1) // 4


def logits_from_param_vector(x, param_vector, hidden_size):
    idx = 0
    fc1_weight_size = hidden_size * 2
    fc1_weight = param_vector[idx:idx + fc1_weight_size].view(hidden_size, 2)
    idx += fc1_weight_size
    fc1_bias = param_vector[idx:idx + hidden_size]
    idx += hidden_size
    fc2_weight = param_vector[idx:idx + hidden_size].view(1, hidden_size)
    idx += hidden_size
    fc2_bias = param_vector[idx:idx + 1]
    hidden = torch.relu(x @ fc1_weight.t() + fc1_bias)
    return hidden @ fc2_weight.t() + fc2_bias


def compute_boundary(param_vector, hidden_size, grid_resolution):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_resolution),
        np.linspace(y_min, y_max, grid_resolution),
    )
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    params = torch.tensor(param_vector, dtype=torch.float32)
    with torch.no_grad():
        grid_logits = logits_from_param_vector(grid_points, params, hidden_size)
        grid_pred = (torch.sigmoid(grid_logits) >= 0.5).long().squeeze(1).numpy()
        logits = logits_from_param_vector(XOR_DATA, params, hidden_size)
        loss = F.binary_cross_entropy_with_logits(logits, XOR_LABELS).item()
    return xx, yy, grid_pred.reshape(xx.shape), float(loss)


def main():
    parser = argparse.ArgumentParser(description="Export XOR boundary snapshots from curve.npz")
    parser.add_argument("--curve-npz", required=True, help="Path to curve.npz")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--t-values",
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated t values (default: 0.0,0.2,0.4,0.6,0.8,1.0)",
    )
    parser.add_argument("--grid-resolution", type=int, default=220, help="Grid resolution (default: 220)")
    args = parser.parse_args()

    data = np.load(args.curve_npz)
    if "ts" not in data or "param_vectors" not in data:
        raise ValueError("curve.npz must contain keys: ts, param_vectors")

    ts = np.asarray(data["ts"], dtype=np.float64)
    param_vectors = np.asarray(data["param_vectors"], dtype=np.float64)
    te_loss = np.asarray(data["te_loss"], dtype=np.float64) if "te_loss" in data else None

    hidden_size = infer_hidden_size(param_vectors.shape[1])
    target_ts = [float(x.strip()) for x in args.t_values.split(",") if x.strip()]
    idxs = [int(np.argmin(np.abs(ts - t))) for t in target_ts]

    os.makedirs(args.output_dir, exist_ok=True)

    xor_x = XOR_DATA[:, 0].numpy()
    xor_y = XOR_DATA[:, 1].numpy()
    xor_labels = XOR_LABELS.squeeze(1).numpy().astype(int)
    point_colors = np.array(["#d62728" if lbl == 0 else "#2ca02c" for lbl in xor_labels])

    rows = []
    for idx in idxs:
        xx, yy, z, fallback_loss = compute_boundary(param_vectors[idx], hidden_size, args.grid_resolution)
        loss = float(te_loss[idx]) if te_loss is not None else fallback_loss

        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5], colors=["#ffcccc", "#ccffcc"], alpha=0.95)
        ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=1.8)
        ax.scatter(xor_x, xor_y, c=point_colors, s=120, edgecolors="black", linewidths=1.2)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"t={ts[idx]:.2f} | loss={loss:.4f}")
        fig.tight_layout()

        file_name = f"snapshot_t{ts[idx]:.2f}".replace(".", "p") + ".png"
        path = os.path.join(args.output_dir, file_name)
        fig.savefig(path, dpi=180)
        plt.close(fig)
        rows.append((path, idx, float(ts[idx]), loss))

    fig, axes = plt.subplots(1, len(idxs), figsize=(3.3 * len(idxs), 3.4), constrained_layout=True)
    if len(idxs) == 1:
        axes = [axes]
    for ax, idx in zip(axes, idxs):
        xx, yy, z, fallback_loss = compute_boundary(param_vectors[idx], hidden_size, 180)
        loss = float(te_loss[idx]) if te_loss is not None else fallback_loss
        ax.contourf(xx, yy, z, levels=[-0.5, 0.5, 1.5], colors=["#ffcccc", "#ccffcc"], alpha=0.95)
        ax.contour(xx, yy, z, levels=[0.5], colors="black", linewidths=1.5)
        ax.scatter(xor_x, xor_y, c=point_colors, s=65, edgecolors="black", linewidths=0.9)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"t={ts[idx]:.1f}\nloss={loss:.3f}", fontsize=10)

    strip_name = "snapshots_strip_" + "_".join(f"{ts[idx]:.1f}" for idx in idxs)
    strip_name = strip_name.replace(".", "p") + ".png"
    strip_path = os.path.join(args.output_dir, strip_name)
    fig.savefig(strip_path, dpi=220)
    plt.close(fig)

    print(f"SNAPSHOTS_DIR {args.output_dir}")
    for path, idx, t, loss in rows:
        print(f"SNAPSHOT {path} idx={idx} t={t:.4f} loss={loss:.6f}")
    print(f"STRIP {strip_path}")


if __name__ == "__main__":
    main()
