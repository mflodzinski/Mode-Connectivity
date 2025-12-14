"""Evaluate curve and save per-sample predictions and features.

Collects predictions at each t value along the curve, and extracts features
from the trained endpoint checkpoints (seed0 and seed1).
"""
import argparse
import numpy as np
import os
import torch
import sys

# Add lib to path
sys.path.insert(0, '../lib')

from lib.core import setup, checkpoint
from lib.utils.args import ArgumentParserBuilder

# Aliases for backward compatibility
EvalSetup = setup
CheckpointLoader = checkpoint.CheckpointLoader


def extract_features(model, loader, device):
    """Extract penultimate layer features from a standard (non-curve) model.

    Args:
        model: VGG16 model to extract features from
        loader: DataLoader to iterate over
        device: Device to run on

    Returns:
        Dictionary with features, predictions, targets, images
    """
    model.eval()

    all_features = []
    all_preds = []
    all_targets = []
    all_images = []

    # Hook to capture penultimate features
    features_hook = []

    def hook_fn(module, input, output):
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
            features = features_hook[0]

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


def evaluate_curve_predictions(curve_model, loader, device, ts):
    """Evaluate curve model at multiple t values and return predictions.

    Args:
        curve_model: CurveNet model to evaluate
        loader: DataLoader to iterate over
        device: Device to run on
        ts: Array of t values to evaluate at

    Returns:
        Predictions array of shape [num_points, num_samples]
    """
    # Import utils from external repo
    import utils as external_utils

    curve_model.eval()

    num_points = len(ts)
    num_samples = len(loader.dataset)

    predictions = np.zeros((num_points, num_samples), dtype=np.int64)

    t_tensor = torch.FloatTensor([0.0]).to(device)

    for i, t_value in enumerate(ts):
        print(f"  Evaluating t={t_value:.3f} ({i+1}/{num_points})")
        t_tensor.data.fill_(t_value)

        # Update batch norm statistics
        external_utils.update_bn(loader, curve_model, device=device, **{'coeffs_t': t_tensor})

        all_preds = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                outputs = curve_model(images, t_tensor)
                preds = outputs.argmax(dim=1)
                all_preds.append(preds.cpu())

        predictions[i] = torch.cat(all_preds, dim=0).numpy()

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Detailed curve evaluation')

    # Custom arguments specific to this script
    parser.add_argument('--curve-ckpt', type=str, required=True,
                        help='Path to curve checkpoint')
    parser.add_argument('--endpoint0-ckpt', type=str, required=True,
                        help='Path to endpoint 0 checkpoint (seed0)')
    parser.add_argument('--endpoint1-ckpt', type=str, required=True,
                        help='Path to endpoint 1 checkpoint (seed1)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for predictions_detailed.npz')
    parser.add_argument('--use-test', action='store_true',
                        help='Use test set')
    parser.add_argument('--num-points', type=int, default=61,
                        help='Number of points to evaluate (default: 61)')
    parser.add_argument('--curve-type', type=str, default='Bezier',
                        help='Curve type (Bezier, PolyChain, etc.) (default: Bezier)')
    parser.add_argument('--num-bends', type=int, default=3,
                        help='Number of bend points (default: 3)')

    # Standard arguments using ArgumentParserBuilder
    ArgumentParserBuilder.add_model_args(parser)
    ArgumentParserBuilder.add_dataset_args(parser)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Setup using lib modules
    EvalSetup.add_external_path()
    device = EvalSetup.get_device()

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    loaders, num_classes = EvalSetup.load_data(
        args.dataset, args.data_path, args.batch_size,
        args.num_workers, args.transform, args.use_test,
        shuffle_train=False
    )

    test_loader = loaders['test']
    architecture = EvalSetup.get_architecture(args.model)

    # Load models using lib
    loader = CheckpointLoader(device)

    # Load curve model
    print(f"\nLoading curve model from {args.curve_ckpt}...")
    curve_model = EvalSetup.create_curve_model(
        architecture, num_classes, args.curve_type, args.num_bends, device
    )
    loader.load_into_model(curve_model, args.curve_ckpt)

    # Load endpoint models
    print(f"\nLoading endpoint 0 from {args.endpoint0_ckpt}...")
    model_t0 = EvalSetup.create_standard_model(architecture, num_classes, device)
    loader.load_into_model(model_t0, args.endpoint0_ckpt)

    print(f"Loading endpoint 1 from {args.endpoint1_ckpt}...")
    model_t1 = EvalSetup.create_standard_model(architecture, num_classes, device)
    loader.load_into_model(model_t1, args.endpoint1_ckpt)

    # Extract features from endpoints
    print(f"\nExtracting features from endpoints...")
    print("Endpoint 0 (t=0):")
    endpoint0_data = extract_features(model_t0, test_loader, device)

    print("Endpoint 1 (t=1):")
    endpoint1_data = extract_features(model_t1, test_loader, device)

    # Verify consistency
    assert np.array_equal(endpoint0_data['targets'], endpoint1_data['targets']), \
        "Targets don't match between endpoints!"
    assert np.allclose(endpoint0_data['images'], endpoint1_data['images']), \
        "Images don't match between endpoints!"

    targets = endpoint0_data['targets']
    images = endpoint0_data['images']
    features_t0 = endpoint0_data['features']
    features_t1 = endpoint1_data['features']

    # Collect predictions along curve
    print(f"\nCollecting predictions along curve at {args.num_points} points...")
    ts = np.linspace(0.0, 1.0, args.num_points)
    predictions = evaluate_curve_predictions(curve_model, test_loader, device, ts)

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
    print(f"  Number of samples: {len(targets)}")
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
