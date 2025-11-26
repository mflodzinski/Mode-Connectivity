"""Compare predictions and per-class performance between two checkpoints.

This script analyzes whether two independently trained models:
- Make different predictions on the same inputs
- Have different per-class accuracies
- Agree or disagree on correct/incorrect predictions
"""
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add external repo to path
sys.path.insert(0, 'external/dnn-mode-connectivity')

import data
import models
import curves


def load_model(checkpoint_path, model_name, num_classes=10):
    """Load a checkpoint and return the model."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_cfg = getattr(models, model_name)
    model = model_cfg.base(num_classes=num_classes, **model_cfg.kwargs)
    model.load_state_dict(checkpoint['model_state'])

    return model


def evaluate_model(model, loader, device):
    """Evaluate model and return predictions, targets, and logits."""
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    all_logits = []
    all_correct = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())
            all_logits.append(logits.cpu())
            all_correct.append((preds == targets).cpu())

    return {
        'preds': torch.cat(all_preds).numpy(),
        'targets': torch.cat(all_targets).numpy(),
        'logits': torch.cat(all_logits).numpy(),
        'correct': torch.cat(all_correct).numpy()
    }


def compute_per_class_accuracy(preds, targets, num_classes=10):
    """Compute per-class accuracy."""
    per_class_acc = []
    per_class_count = []

    for class_idx in range(num_classes):
        mask = targets == class_idx
        if mask.sum() > 0:
            acc = (preds[mask] == targets[mask]).mean()
            per_class_acc.append(acc * 100)
            per_class_count.append(mask.sum())
        else:
            per_class_acc.append(0.0)
            per_class_count.append(0)

    return np.array(per_class_acc), np.array(per_class_count)


def compute_confusion_matrix(preds, targets, num_classes=10):
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, target in zip(preds, targets):
        cm[target, pred] += 1
    return cm


def analyze_agreement(results1, results2):
    """Analyze agreement between two models."""
    preds1 = results1['preds']
    preds2 = results2['preds']
    targets = results1['targets']
    correct1 = results1['correct']
    correct2 = results2['correct']

    # Overall agreement
    agree = (preds1 == preds2)
    agreement_rate = agree.mean() * 100

    # Agreement on correct vs incorrect
    both_correct = correct1 & correct2
    both_incorrect = (~correct1) & (~correct2)
    model1_only = correct1 & (~correct2)
    model2_only = (~correct1) & correct2

    # Where they disagree
    disagree_indices = np.where(~agree)[0]

    return {
        'agreement_rate': agreement_rate,
        'both_correct': both_correct.sum(),
        'both_incorrect': both_incorrect.sum(),
        'model1_only_correct': model1_only.sum(),
        'model2_only_correct': model2_only.sum(),
        'disagree_indices': disagree_indices,
        'total_samples': len(targets)
    }


def plot_per_class_comparison(acc1, acc2, class_names, output_path):
    """Plot per-class accuracy comparison."""
    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, acc1, width, label='Seed 0', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, acc2, width, label='Seed 1', alpha=0.8, color='#ff7f0e')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 105])

    # Add difference annotations
    for i in range(len(class_names)):
        diff = acc2[i] - acc1[i]
        y_pos = max(acc1[i], acc2[i]) + 2
        color = 'green' if abs(diff) < 5 else 'orange' if abs(diff) < 10 else 'red'
        ax.text(i, y_pos, f'{diff:+.1f}', ha='center', va='bottom',
                fontsize=8, color=color, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Per-class comparison plot saved to: {output_path}")
    plt.close()


def plot_confusion_matrices(cm1, cm2, class_names, output_path):
    """Plot confusion matrices side by side."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Normalize confusion matrices
    cm1_norm = cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
    cm2_norm = cm2.astype('float') / cm2.sum(axis=1)[:, np.newaxis]

    # Plot first confusion matrix
    sns.heatmap(cm1_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax1, cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    ax1.set_title('Seed 0 Confusion Matrix', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Class', fontsize=12)
    ax1.set_ylabel('True Class', fontsize=12)

    # Plot second confusion matrix
    sns.heatmap(cm2_norm, annot=True, fmt='.2f', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax2, cbar_kws={'label': 'Accuracy'}, vmin=0, vmax=1)
    ax2.set_title('Seed 1 Confusion Matrix', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Class', fontsize=12)
    ax2.set_ylabel('True Class', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to: {output_path}")
    plt.close()


def plot_agreement_analysis(agreement_info, output_path):
    """Plot agreement analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Both\nCorrect', 'Both\nIncorrect', 'Only Seed 0\nCorrect', 'Only Seed 1\nCorrect']
    values = [
        agreement_info['both_correct'],
        agreement_info['both_incorrect'],
        agreement_info['model1_only_correct'],
        agreement_info['model2_only_correct']
    ]
    percentages = [v / agreement_info['total_samples'] * 100 for v in values]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']

    bars = ax.bar(categories, percentages, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val, pct in zip(bars, values, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Percentage of Samples (%)', fontsize=12)
    ax.set_title(f'Prediction Agreement Analysis\n(Overall Agreement: {agreement_info["agreement_rate"]:.2f}%)',
                 fontsize=14, fontweight='bold')
    ax.set_ylim([0, max(percentages) * 1.2])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Agreement analysis plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare two model checkpoints')
    parser.add_argument('--checkpoint1', type=str, required=True,
                        help='Path to first checkpoint')
    parser.add_argument('--checkpoint2', type=str, required=True,
                        help='Path to second checkpoint')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Dataset name')
    parser.add_argument('--data_path', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--model', type=str, default='VGG16',
                        help='Model architecture')
    parser.add_argument('--transform', type=str, default='VGG',
                        help='Data transformation')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                        help='Output directory for results')
    parser.add_argument('--name1', type=str, default='Seed 0',
                        help='Name for first model')
    parser.add_argument('--name2', type=str, default='Seed 1',
                        help='Name for second model')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
        use_test=True,
        shuffle_train=False
    )
    test_loader = loaders['test']

    # Get class names
    if args.dataset == 'CIFAR10':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        class_names = [f'Class {i}' for i in range(num_classes)]

    # Load models
    print(f"\nLoading {args.name1} from {args.checkpoint1}...")
    model1 = load_model(args.checkpoint1, args.model, num_classes)

    print(f"Loading {args.name2} from {args.checkpoint2}...")
    model2 = load_model(args.checkpoint2, args.model, num_classes)

    # Evaluate both models
    print(f"\nEvaluating {args.name1}...")
    results1 = evaluate_model(model1, test_loader, device)
    acc1 = results1['correct'].mean() * 100

    print(f"Evaluating {args.name2}...")
    results2 = evaluate_model(model2, test_loader, device)
    acc2 = results2['correct'].mean() * 100

    print(f"\n{args.name1} Test Accuracy: {acc1:.2f}%")
    print(f"{args.name2} Test Accuracy: {acc2:.2f}%")

    # Compute per-class accuracies
    print("\nComputing per-class accuracies...")
    per_class_acc1, per_class_count1 = compute_per_class_accuracy(
        results1['preds'], results1['targets'], num_classes)
    per_class_acc2, per_class_count2 = compute_per_class_accuracy(
        results2['preds'], results2['targets'], num_classes)

    # Compute confusion matrices
    print("Computing confusion matrices...")
    cm1 = compute_confusion_matrix(results1['preds'], results1['targets'], num_classes)
    cm2 = compute_confusion_matrix(results2['preds'], results2['targets'], num_classes)

    # Analyze agreement
    print("Analyzing prediction agreement...")
    agreement_info = analyze_agreement(results1, results2)

    # Print results
    print("\n" + "="*70)
    print("CHECKPOINT COMPARISON SUMMARY")
    print("="*70)

    print(f"\nOverall Performance:")
    print(f"  {args.name1} accuracy: {acc1:.2f}%")
    print(f"  {args.name2} accuracy: {acc2:.2f}%")
    print(f"  Difference: {acc2 - acc1:+.2f}%")

    print(f"\nPer-Class Accuracy:")
    print(f"  {'Class':<12} {args.name1:>10} {args.name2:>10} {'Diff':>10} {'Count':>8}")
    print(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")
    for i, name in enumerate(class_names):
        diff = per_class_acc2[i] - per_class_acc1[i]
        print(f"  {name:<12} {per_class_acc1[i]:>9.2f}% {per_class_acc2[i]:>9.2f}% "
              f"{diff:>+9.2f}% {per_class_count1[i]:>8}")

    print(f"\nPrediction Agreement:")
    print(f"  Overall agreement rate: {agreement_info['agreement_rate']:.2f}%")
    print(f"  Both models correct: {agreement_info['both_correct']} "
          f"({agreement_info['both_correct']/agreement_info['total_samples']*100:.2f}%)")
    print(f"  Both models incorrect: {agreement_info['both_incorrect']} "
          f"({agreement_info['both_incorrect']/agreement_info['total_samples']*100:.2f}%)")
    print(f"  Only {args.name1} correct: {agreement_info['model1_only_correct']} "
          f"({agreement_info['model1_only_correct']/agreement_info['total_samples']*100:.2f}%)")
    print(f"  Only {args.name2} correct: {agreement_info['model2_only_correct']} "
          f"({agreement_info['model2_only_correct']/agreement_info['total_samples']*100:.2f}%)")

    # Find classes with biggest differences
    print(f"\nBiggest Per-Class Differences:")
    diffs = per_class_acc2 - per_class_acc1
    sorted_indices = np.argsort(np.abs(diffs))[::-1]
    for idx in sorted_indices[:5]:
        print(f"  {class_names[idx]:<12}: {diffs[idx]:+.2f}% "
              f"({args.name1}: {per_class_acc1[idx]:.2f}%, {args.name2}: {per_class_acc2[idx]:.2f}%)")

    print("="*70)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_per_class_comparison(
        per_class_acc1, per_class_acc2, class_names,
        output_dir / 'per_class_comparison.png'
    )

    plot_confusion_matrices(
        cm1, cm2, class_names,
        output_dir / 'confusion_matrices.png'
    )

    plot_agreement_analysis(
        agreement_info,
        output_dir / 'agreement_analysis.png'
    )

    # Save numerical results
    summary_path = output_dir / 'comparison_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("CHECKPOINT COMPARISON SUMMARY\n")
        f.write("="*70 + "\n\n")

        f.write(f"Checkpoint 1: {args.checkpoint1}\n")
        f.write(f"Checkpoint 2: {args.checkpoint2}\n\n")

        f.write(f"Overall Performance:\n")
        f.write(f"  {args.name1} accuracy: {acc1:.2f}%\n")
        f.write(f"  {args.name2} accuracy: {acc2:.2f}%\n")
        f.write(f"  Difference: {acc2 - acc1:+.2f}%\n\n")

        f.write(f"Per-Class Accuracy:\n")
        f.write(f"  {'Class':<12} {args.name1:>10} {args.name2:>10} {'Diff':>10} {'Count':>8}\n")
        f.write(f"  {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*8}\n")
        for i, name in enumerate(class_names):
            diff = per_class_acc2[i] - per_class_acc1[i]
            f.write(f"  {name:<12} {per_class_acc1[i]:>9.2f}% {per_class_acc2[i]:>9.2f}% "
                    f"{diff:>+9.2f}% {per_class_count1[i]:>8}\n")

        f.write(f"\nPrediction Agreement:\n")
        f.write(f"  Overall agreement rate: {agreement_info['agreement_rate']:.2f}%\n")
        f.write(f"  Both models correct: {agreement_info['both_correct']} "
                f"({agreement_info['both_correct']/agreement_info['total_samples']*100:.2f}%)\n")
        f.write(f"  Both models incorrect: {agreement_info['both_incorrect']} "
                f"({agreement_info['both_incorrect']/agreement_info['total_samples']*100:.2f}%)\n")
        f.write(f"  Only {args.name1} correct: {agreement_info['model1_only_correct']} "
                f"({agreement_info['model1_only_correct']/agreement_info['total_samples']*100:.2f}%)\n")
        f.write(f"  Only {args.name2} correct: {agreement_info['model2_only_correct']} "
                f"({agreement_info['model2_only_correct']/agreement_info['total_samples']*100:.2f}%)\n")

        f.write("="*70 + "\n")

    print(f"\nSummary saved to: {summary_path}")
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
