"""Evaluate curve and save per-sample predictions and features.

This script extends eval_curve.py to save detailed per-sample information:
- Predictions at each t value
- Penultimate layer features at t=0 and t=1 (endpoints)
- Ground truth labels
- Raw images
"""
import argparse
import numpy as np
import os
import torch
import torch.nn.functional as F
import sys

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')

import data
import models
import curves


def extract_features(model, loader, device):
    """Extract penultimate layer features and predictions from model.

    For VGG16, penultimate features are the 512-d vector before the final linear layer.
    """
    model.eval()

    all_features = []
    all_preds = []
    all_targets = []
    all_images = []

    # Hook to capture penultimate features
    features_hook = []

    def hook_fn(module, input, output):
        # Capture output of second-to-last ReLU in classifier
        features_hook.append(output.detach())

    # Register hook on the classifier's penultimate ReLU (index 5)
    hook = model.classifier[5].register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)

            # Forward pass (hook will capture features)
            features_hook.clear()
            outputs = model(images)

            preds = outputs.argmax(dim=1)
            features = features_hook[0]  # Get captured features

            all_features.append(features.cpu())
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_images.append(images.cpu())

    hook.remove()

    return {
        'features': torch.cat(all_features, dim=0).numpy(),
        'predictions': torch.cat(all_preds, dim=0).numpy(),
        'targets': torch.cat(all_targets, dim=0).numpy(),
        'images': torch.cat(all_images, dim=0).numpy()
    }


def evaluate_at_t(model, loader, device, t_value):
    """Evaluate model at specific t value and return predictions."""
    model.eval()

    all_preds = []

    t = torch.FloatTensor([t_value]).to(device)

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images, t)
            preds = outputs.argmax(dim=1)
            all_preds.append(preds.cpu())

    return torch.cat(all_preds, dim=0).numpy()


def main():
    parser = argparse.ArgumentParser(description='Detailed curve evaluation with predictions and features')
    parser.add_argument('--curve_ckpt', type=str, required=True,
                        help='Path to curve checkpoint')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for predictions_detailed.npz')
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--model', type=str, default='VGG16')
    parser.add_argument('--transform', type=str, default='VGG')
    parser.add_argument('--curve', type=str, default='Bezier')
    parser.add_argument('--num_bends', type=int, default=3)
    parser.add_argument('--num_points', type=int, default=61,
                        help='Number of evaluation points along curve')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_test', action='store_true')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        args.transform,
        args.use_test,
        shuffle_train=False
    )

    test_loader = loaders['test']

    # Load curve model
    print(f"\nLoading curve model from {args.curve_ckpt}...")
    architecture = getattr(models, args.model)
    curve_cls = getattr(curves, args.curve)
    model = curves.CurveNet(
        num_classes,
        curve_cls,
        architecture.curve,
        args.num_bends,
        architecture_kwargs=architecture.kwargs,
    )
    model.to(device)

    checkpoint = torch.load(args.curve_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state'])

    print(f"\nExtracting features and predictions at endpoints...")

    # Extract features at t=0 (endpoint 0)
    print("\nEndpoint 0 (t=0):")
    t0 = torch.FloatTensor([0.0]).to(device)
    model_t0 = model.export_base(t0)
    endpoint0_data = extract_features(model_t0, test_loader, device)

    # Extract features at t=1 (endpoint 1)
    print("\nEndpoint 1 (t=1):")
    t1 = torch.FloatTensor([1.0]).to(device)
    model_t1 = model.export_base(t1)
    endpoint1_data = extract_features(model_t1, test_loader, device)

    # Verify targets match
    assert np.array_equal(endpoint0_data['targets'], endpoint1_data['targets']), \
        "Targets don't match between endpoints!"

    # Verify images match
    assert np.allclose(endpoint0_data['images'], endpoint1_data['images']), \
        "Images don't match between endpoints!"

    targets = endpoint0_data['targets']
    images = endpoint0_data['images']
    features_t0 = endpoint0_data['features']
    features_t1 = endpoint1_data['features']

    print(f"\nCollecting predictions along curve at {args.num_points} points...")

    # Collect predictions at all t values
    ts = np.linspace(0.0, 1.0, args.num_points)
    num_samples = len(targets)

    predictions = np.zeros((args.num_points, num_samples), dtype=np.int64)

    for i, t_value in enumerate(ts):
        predictions[i] = evaluate_at_t(model, test_loader, device, t_value)

    # Save all data
    print(f"\nSaving results to {args.output}...")
    np.savez_compressed(
        args.output,
        predictions=predictions,  # [num_points, num_samples]
        targets=targets,  # [num_samples]
        features_t0=features_t0,  # [num_samples, 512]
        features_t1=features_t1,  # [num_samples, 512]
        images=images,  # [num_samples, 3, 32, 32]
        ts=ts,  # [num_points]
    )

    print("\nDataset info:")
    print(f"  Number of samples: {num_samples}")
    print(f"  Number of evaluation points: {args.num_points}")
    print(f"  Feature dimension: {features_t0.shape[1]}")
    print(f"  Image shape: {images.shape[1:]}")

    print(f"\nSaved shapes:")
    print(f"  predictions: {predictions.shape}")
    print(f"  targets: {targets.shape}")
    print(f"  features_t0: {features_t0.shape}")
    print(f"  features_t1: {features_t1.shape}")
    print(f"  images: {images.shape}")
    print(f"  ts: {ts.shape}")

    print(f"\nDone! Results saved to {args.output}")


if __name__ == "__main__":
    main()
