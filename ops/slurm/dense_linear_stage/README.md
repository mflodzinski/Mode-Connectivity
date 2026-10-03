# Final-endpoint alignment benchmark

This directory launches the larger-network experiment reported in Figure 4,
Appendix Figure A.2, and Tables A.1--A.3. The `dense_linear_stage` name is kept
for import and result compatibility, but the active pipeline accepts exactly
one final checkpoint per endpoint; the historical 12-by-12 cross-stage mode is
archived.

For each architecture, the benchmark evaluates three disjoint endpoint pairs:
`(0,1)`, `(2,3)`, and `(4,5)`. Each pair independently calibrates all optimized
methods on frozen validation splits, then evaluates six paths: raw, weight
matching (WM), WM plus scale refinement, Sinkhorn, joint Sinkhorn-scale, and
Sinkhorn followed by scale refinement.

Test data are not used for hyperparameter selection. Alignment fitting uses the
complete endpoint-training split. Standard reports use a deterministic,
class-balanced 10,000-example training subset and the complete test set; the
post-hoc full-training evaluation covers all 45,000 CIFAR-10 or 55,000
Fashion-MNIST training examples.

## Preview and submit

Always inspect the generated Slurm commands first:

```bash
bash ops/slurm/dense_linear_stage/submit_all.sh --dry-run \
  --config-name dense_linear_stage/final_vgg11
```

Submit by omitting `--dry-run`. Available configs are `final_vgg11`,
`final_vgg13`, `final_vgg16`, `final_vgg19`, and `final_fashion_mnist`.
Submissions are resumable, and the protocol hash prevents incompatible reuse
of an existing result root.

## Monitor

```bash
bash ops/slurm/dense_linear_stage/status.sh \
  results/final_alignment_vgg11_cifar10
```

## Evaluate the complete training split

After the standard reports finish, reuse the frozen alignments for the paper's
full-training-split measurement:

```bash
bash ops/slurm/dense_linear_stage/submit_full_train.sh \
  results/final_alignment_vgg11_cifar10 \
  results/final_alignment_vgg13_cifar10 \
  results/final_alignment_vgg16_cifar10 \
  results/final_alignment_vgg19_cifar10_atol5e5 \
  results/final_alignment_fashion_mnist
```

The scheduler workers `run_cpu.sh`, `run_gpu.sh`, and `run_full_train.sh` are
internal entry points. See [../../../REPRODUCIBILITY.md](../../../REPRODUCIBILITY.md)
for endpoint-training commands and the complete paper-to-code map.
