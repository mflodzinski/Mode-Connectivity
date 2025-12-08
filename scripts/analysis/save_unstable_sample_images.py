"""Save images of samples with most prediction changes along the curve.

This script extracts and saves images of the most unstable samples - those whose
predictions change most frequently along the mode connectivity curve.
"""

import argparse
import numpy as np
import os
from PIL import Image


def denormalize_cifar10(img):
    """Denormalize CIFAR-10 image.

    Args:
        img: Image array in CHW format, normalized

    Returns:
        Image array in HWC format, denormalized to [0, 255] uint8
    """
    # CIFAR-10 normalization stats
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])

    # Convert from CHW to HWC
    img = img.transpose(1, 2, 0)

    # Denormalize
    img = img * std + mean

    # Clip and convert to uint8
    img = np.clip(img * 255, 0, 255).astype(np.uint8)

    return img


def main():
    parser = argparse.ArgumentParser(
        description='Save images of most unstable samples'
    )
    parser.add_argument('--predictions-file', type=str, required=True,
                        help='Path to predictions_detailed.npz file')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save sample images')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Number of top unstable samples to save (default: 20)')

    args = parser.parse_args()

    print("="*70)
    print("SAVING UNSTABLE SAMPLE IMAGES")
    print("="*70)

    # Load data
    print(f"\nLoading predictions from: {args.predictions_file}")
    data = np.load(args.predictions_file)

    preds = data['predictions']  # Shape: (num_t_points, num_samples)
    targets = data['targets']
    images = data['images']

    print(f"  Predictions shape: {preds.shape}")
    print(f"  Images shape: {images.shape}")

    # CIFAR-10 class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # Calculate changes for each sample
    print(f"\nCalculating prediction changes for all samples...")
    changes_per_sample = []
    for i in range(preds.shape[1]):
        sample_preds = preds[:, i]
        n_changes = np.sum(sample_preds[:-1] != sample_preds[1:])
        changes_per_sample.append(n_changes)

    changes_per_sample = np.array(changes_per_sample)

    # Get top K most unstable samples
    top_indices = np.argsort(changes_per_sample)[::-1][:args.top_k]

    print(f"\nTop {args.top_k} samples with most prediction changes:")
    print(f"{'Rank':<6} {'Sample':<8} {'Changes':<10} {'True Label':<15}")
    print("-"*50)
    for rank, idx in enumerate(top_indices, 1):
        n_changes = changes_per_sample[idx]
        true_label = class_names[targets[idx]]
        print(f"{rank:<6} {idx:<8} {n_changes:<10} {true_label:<15}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save each image
    print(f"\nSaving images to: {args.output_dir}")
    print("-"*70)

    for rank, sample_idx in enumerate(top_indices, 1):
        # Get image
        img = images[sample_idx]  # CHW format, normalized

        # Denormalize
        img_denorm = denormalize_cifar10(img)

        # Create PIL Image
        pil_img = Image.fromarray(img_denorm)

        # Upscale with LANCZOS for smoother interpolation
        pil_img_large = pil_img.resize((256, 256), Image.LANCZOS)

        # Get metadata
        n_changes = changes_per_sample[sample_idx]
        true_label = class_names[targets[sample_idx]]
        start_pred = class_names[preds[0, sample_idx]]
        end_pred = class_names[preds[-1, sample_idx]]

        # Save upscaled version with metadata in filename
        filename = f"rank{rank:02d}_sample{sample_idx:05d}_{n_changes}changes_{true_label}.png"
        filepath = os.path.join(args.output_dir, filename)
        pil_img_large.save(filepath)

        print(f"  {rank:2d}. Saved {filename}")
        print(f"      True: {true_label:12s} | {n_changes} changes | {start_pred} → {end_pred}")

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"\nSaved {args.top_k} images to: {args.output_dir}")
    print(f"  - All images upscaled to 256×256 (LANCZOS interpolation from 32×32 originals)")
    print(f"\nNote: CIFAR-10 images are inherently 32×32 pixels.")
    print(f"Images have been upscaled for visibility, but resolution is limited by the dataset.")
    print("="*70)


if __name__ == "__main__":
    main()
