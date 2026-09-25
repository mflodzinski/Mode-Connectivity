# Dense linear training-stage experiment

This family evaluates the complete ordered 12-by-12 checkpoint matrix for
three seed pairs. For every matrix cell, seed pair `(0,1)` compares all six
methods and the configured hyperparameter candidates. Each cell freezes both
the best strict permutation-only method (`WM` or `Sinkhorn`) and the best
method overall. The union of those two pipelines is then fitted to seed pairs
`(2,3)` and `(4,5)`. Alignment uses the complete endpoint-training split
(45,000 CIFAR-10 or 55,000 Fashion-MNIST examples), selection never reads test
data, and final profiles use a frozen stratified 10,000-example training subset
and the complete official 10,000-example test split. The reporting subset is
used only for final metrics and never replaces the full alignment-fit split.

Before fitting, an audited reuse task scans the previous experiment roots.
It imports data-independent WM artifacts only when model settings, WM seed,
WM sweeps, direction, endpoint identities, and endpoint SHA-256 hashes match.
For VGG11, the prior 4-by-4 Sinkhorn grid supplies cell-specific
validation-only hyperparameter priors for its 16 covered cells. Those cells
still refit Sinkhorn/scale on the new full training split. Legacy Sinkhorn
artifacts and sampled train/test profiles are deliberately not treated as
final results because they used the earlier 5,000/2,000-example protocol.
Every decision is recorded in `reuse_inventory.json` and
`hyperparameter_priors.json`.

The historical VGG11 grid also justifies two global settings. Sinkhorn
`lr=0.05, tau=1.5, l=1` won 14 of 16 cells; the two `tau=1` exceptions had
validation-barrier advantages of only 0.0004 and 0.0020. Fixed-permutation
scale refinement with `lr=0.05` and penalty `1e-4` won all 16 cells. Those two
settings are frozen globally, so their per-cell candidate searches are skipped.
WM+scale and joint Sinkhorn+scale have no matching historical sweep and remain
calibrated. The Fashion-MNIST history contains only one setting rather than a
candidate grid, so it is not treated as winner-frequency evidence.

Start with five-cell resource pilots. They cover early–early, both
early/final directions, a middle diagonal, and final–final:

```bash
bash ops/slurm/dense_linear_stage/submit_pilot.sh \
  --config-name dense_linear_stage/vgg11

bash ops/slurm/dense_linear_stage/submit_pilot.sh \
  --config-name dense_linear_stage/fashion_mnist
```

After inspecting `sacct`, adjust the resource entries and submit the remaining
resumable graph. Completed pilot cells are reused. Scheduler resources
and chunk sizes are deliberately excluded from the scientific protocol hash,
so they can be tightened in the same root after calibration:

```bash
bash ops/slurm/dense_linear_stage/recommend_resources.sh \
  results/dense_linear_stage_vgg11
```

```bash
bash ops/slurm/dense_linear_stage/submit_main.sh \
  --config-name dense_linear_stage/vgg11

bash ops/slurm/dense_linear_stage/submit_main.sh \
  --config-name dense_linear_stage/fashion_mnist
```

Use `--dry-run` to inspect every `sbatch` command. The default array
concurrency is three because the requested DAIC resource is an A40.
The measured VGG11 pilot led to 20-minute base and 15-minute branch requests,
with 4 GB host memory and no DataLoader subprocess for fitting. Calibration
evaluation receives 30 minutes, and a bundled two-cell held-out fit/evaluation
receives one hour. Fashion-MNIST bundles more cells per job and uses 20–45
minute requests. These are
initial estimates; run the resource recommender after the pilot and before the
full submission.

Check durable task state with:

```bash
bash ops/slurm/dense_linear_stage/status.sh results/dense_linear_stage_vgg11
```

Calibration uses seeds `(0,1)` only. It selects the best permutation-only
method and the best method overall, together with their settings, separately
for every ordered epoch pair. The selections are frozen in
`hyperparameters.json` and `selections.json`; no grid search is repeated for
seed pairs `(2,3)` and `(4,5)`. Reports show every method for pair `(0,1)` and
both frozen choices as mean ± standard deviation across all three seed pairs.
The predeclared selection criterion is validation-loss `B_worse`, with
validation-loss `B_chord`, maximum loss, mean loss, and fixed method order as
tie breakers. The permutation-only choice is restricted to `{WM, Sinkhorn}`;
the overall choice ranges over all six methods. Test loss and test error are
reported outcomes, never selection criteria.
