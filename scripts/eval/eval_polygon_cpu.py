"""
Evaluate polygon chain or symmetry plane with automatic device detection.
Supports CUDA, MPS (Apple Silicon), and CPU fallback.
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')

import data
import models
import curves
import utils


def get_device():
    """Get best available device: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser(description='Evaluate PolyChain curve (multi-device support)')
    parser.add_argument('--dir', type=str, required=True, help='output directory')
    parser.add_argument('--num_points', type=int, default=61, help='number of evaluation points')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--use_test', action='store_true', help='use test set')
    parser.add_argument('--transform', type=str, required=True, help='transform type')
    parser.add_argument('--data_path', type=str, required=True, help='path to data')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--model', type=str, required=True, help='model architecture')
    parser.add_argument('--curve', type=str, required=True, help='curve type')
    parser.add_argument('--num_bends', type=int, required=True, help='number of bends')
    parser.add_argument('--ckpt', type=str, required=True, help='checkpoint path')
    
    args = parser.parse_args()
    
    # Auto-detect best device
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs(args.dir, exist_ok=True)
    
    # Load data
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        shuffle_train=False
    )
    
    # Build model
    architecture = getattr(models, args.model)
    curve_class = getattr(curves, args.curve)
    model = curves.CurveNet(
        num_classes,
        curve_class,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state'])
    
    criterion = F.cross_entropy
    
    # Evaluation
    T = args.num_points
    ts = np.linspace(0.0, 1.0, T)
    
    tr_loss = np.zeros(T)
    tr_acc = np.zeros(T)
    te_loss = np.zeros(T)
    te_acc = np.zeros(T)
    tr_err = np.zeros(T)
    te_err = np.zeros(T)
    
    print(f"\nEvaluating at {T} points along the curve...")
    print("=" * 80)
    
    for i, t_value in enumerate(ts):
        t = torch.FloatTensor([t_value]).to(device)
        
        # Update batch normalization statistics
        utils.update_bn(loaders['train'], model, device=device, t=t)
        
        # Evaluate on train set
        train_res = utils.test(loaders['train'], model, criterion, device=device, t=t)
        tr_loss[i] = train_res['loss']
        tr_acc[i] = train_res['accuracy']
        tr_err[i] = 100.0 - tr_acc[i]
        
        # Evaluate on test set
        test_res = utils.test(loaders['test'], model, criterion, device=device, t=t)
        te_loss[i] = test_res['loss']
        te_acc[i] = test_res['accuracy']
        te_err[i] = 100.0 - te_acc[i]
        
        if (i + 1) % 10 == 0 or i == 0 or i == T - 1:
            print(f"  [{i+1:2d}/{T}] t={t_value:.3f} | "
                  f"Train: {tr_loss[i]:.4f}/{tr_acc[i]:.2f}% | "
                  f"Test: {te_loss[i]:.4f}/{te_acc[i]:.2f}%")
    
    # Save results
    output_file = os.path.join(args.dir, 'curve.npz')
    np.savez(
        output_file,
        ts=ts,
        tr_loss=tr_loss,
        tr_acc=tr_acc,
        tr_err=tr_err,
        te_loss=te_loss,
        te_acc=te_acc,
        te_err=te_err,
    )
    
    print("=" * 80)
    print("\nEVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Train - Max loss: {np.max(tr_loss):.4f} | Max error: {np.max(tr_err):.2f}% | Min acc: {np.min(tr_acc):.2f}%")
    print(f"  Test  - Max loss: {np.max(te_loss):.4f} | Max error: {np.max(te_err):.2f}% | Min acc: {np.min(te_acc):.2f}%")
    print("=" * 80)


if __name__ == '__main__':
    main()
