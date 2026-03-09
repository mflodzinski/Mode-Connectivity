# VGG16/CIFAR10 Sinkhorn+Scale Prototype

This prototype implements a VGG16-only, endpoint-only extension of Sinkhorn re-basin for comparing:

- no alignment
- Sinkhorn permutation-only alignment
- Sinkhorn permutation + positive diagonal scaling

## Formulas

For each hidden layer `l`, the trainable alignment uses

- `S_l = sinkhorn(Z_l / tau)`
- `D_l = diag(exp(u_l))`
- `M_l = S_l D_l`

The transformed representative of model `B` is built layerwise:

- hidden output transport uses `M_l`
- next-layer input transport uses `S_l D_l^{-1}`

The input-side transport is the code equivalent of right-multiplying by
`D_l^{-1} S_l^T`, which matches the permutation-only baseline convention
without inverting a dense soft matrix.

## Objective

Only the interpolation barrier objective is optimized:

- `J_bar = mean_{alpha in [0.25, 0.5, 0.75]} L((1-alpha) theta_A + alpha theta_B')`
- `J = J_bar + lambda_scale * sum_l ||u_l||_2^2`

Excluded on purpose:

- checkpoint or continuation alignment
- REPAIR
- weight-difference penalties
- activation-matching penalties

## Implementation Notes

- Architecture support is intentionally limited to the repo's CIFAR10 VGG16.
- The conv-to-FC transition assumes the final spatial map is `1x1`, which is true for this model on `32x32` CIFAR10 inputs.
- Alignment training uses `torch.func.functional_call` so gradients flow through the interpolated weights.
- Models stay in `eval()` mode during alignment. This disables classifier dropout and keeps the objective deterministic.
- Hard projection is done per layer with Hungarian matching on the learned soft Sinkhorn matrix, while preserving the learned positive scales.

## Scripts

Train the alignment variables:

```bash
python scripts/analysis/train_vgg16_sinkhorn_scale.py
```

Evaluate interpolation curves and write summaries/plots:

```bash
python scripts/analysis/evaluate_vgg16_sinkhorn_scale.py
```

Both scripts default to:

- `results/vgg16/cifar10/endpoints/standard/seed0/checkpoints/checkpoint-200.pt`
- `results/vgg16/cifar10/endpoints/standard/seed1/checkpoints/checkpoint-200.pt`

and write outputs under:

```text
results/vgg16/cifar10/alignment/sinkhorn_scale_prototype/seed0-seed1
```

## Limitations

- V1 is not a generic permutation-spec framework.
- The soft transport can outperform the hard projection; the evaluation explicitly reports whether the hard monomial alignment preserves the gain.
- The prototype has only been wired for VGG16/CIFAR10 and should be treated as a research baseline, not production infrastructure.
