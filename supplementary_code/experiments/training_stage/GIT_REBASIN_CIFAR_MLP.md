# Git Re-Basin CIFAR-10 MLP protocol

This preset reproduces the setup behind Figure 3 of Ainsworth, Hayase, and
Srinivasa (2022), then extends it to a cross-stage matrix.

Official sources pinned while implementing the preset:

- Git Re-Basin repository commit: `ef40098257ab97243930eba737d6dcb8edd5863e`
- Architecture/training: `src/cifar10_mlp_train.py`
- Onset matching/evaluation: `src/cifar10_mlp_barrier_vs_epoch_matching.py`
- Figure metric: `src/cifar10_mlp_barrier_vs_epoch_plot.py`
- AugMax pin: commit `3e5d85d6921a1e519987d33f226bc13f61e04d04`
- Public endpoint artifacts: `cifar10-mlp-weights:v13` (seed 0) and `:v14`
  (seed 1). Both used SGD, peak LR 0.1, batch size 100, and 100 epochs.

The model is 3072-512-512-512-10 with ReLU after each hidden layer and
log-softmax after the output. Training uses all 50,000 training examples,
one-epoch linear warmup from 1e-6 to 0.1, cosine decay, momentum 0.9, and weight
decay 5e-4. The augmentation is RandomSizedCrop with zoom range (0.8, 1.2),
horizontal flip, and rotation in [-30, 30] degrees, followed by division by 255
and ImageNet normalization.

The official JAX/Flax experiment is implemented here in PyTorch so it can use
the project's alignment and Slurm machinery. Layer layout, initializer
distribution, optimizer equation, schedule, augmentation distribution,
preprocessing, epoch count, batch size, and matching/evaluation budgets match.
JAX and PyTorch do not produce the same random samples or bitwise numerical
trajectory from the same integer seed. Results are therefore a protocol
replication, not a byte-for-byte replay of the published checkpoints.

The official code derives initialization, epoch permutations, and per-batch
augmentation keys from each run's seed. Thus the literal implementation does
not hold augmentation draws fixed between seeds 0 and 1, even though the paper
describes the comparison as different initialization and data order. The preset
uses `augmentation_seed_mode: run` to follow the code. Setting it to `shared`
is available as a separate ablation, but is not the Figure 3 reproduction.

The paper's `checkpoint0` is the state after its first completed epoch. This
preset stores true initialization as `epoch_000.pt`; `epoch_001.pt` corresponds
to the paper's `checkpoint0`. Onset outputs record both labels.

The paper's plotted barrier is `max(loss) - (loss_A + loss_B)/2`. That value is
retained as `git_rebasin`. The extension also reports chord and worse-endpoint
barriers, which remain interpretable for cross-stage pairs with unequal endpoint
losses.
