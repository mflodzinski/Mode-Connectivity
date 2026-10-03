# Reproducing the paper

All commands below run from the repository root with `PYTHONPATH=.:src`. Commands prefixed by `poetry run` can equivalently be run inside the Poetry environment. The XOR studies are CPU-friendly; the larger-network experiments are designed for a Slurm cluster with NVIDIA A40-class GPUs and are resumable.

## Paper-to-code map

| Paper result | Experiment entry point and configuration | Retained result source | Figure/table generator |
|---|---|---|---|
| Figure 1; width-2 exhaustive identity/swap result (`6/28` connected, `22/28` obstructed) | `experiments.xor.basin_test`; `configs/experiments/xor/runners/basin_test.yaml` | `results/xor/xor_2h_15seeds/` | `tools/plotting/plot_xor_identity_swap_examples.py` |
| Width-2 minibatch-SGD robustness result | `experiments.xor.train_linear_barriers --train-batch-size 2` | `results/xor_minibatch_sgd/xor_2h_trained_linear_pairs/` | metrics are written by the runner |
| Table 1; exhaustive permutation, Sinkhorn, joint permutation-scale, and frozen-permutation scale refinement | `experiments.xor.train_linear_barriers` followed by `experiments.xor.permutation_scale`; runner/search YAML under `configs/experiments/xor/` | `results/xor/xor_{3,5,7}h_trained_linear_pairs/` and `results/xor/xor_{3,5,7}h_perm_vs_scale/` | aggregate Markdown/JSON from the runner |
| Appendix Figure A.1 | same width-3/5/7 alignment runs as Table 1 | `results/xor/xor_{3,5,7}h_perm_vs_scale/` | `tools/plotting/plot_xor_width_profiles.py` |
| Figure 2; fixed-endpoint nonlinear XOR paths and width comparison | `experiments.xor.curve_fitting` | retained representative panels `figures/unsuccesful_10-14_bezier_3bend.png` and `figures/3hidden_xor_bezier.png`; supporting width-2 runs under `results/xor/xor_2h_*curves*/` | `tools/plotting/compose_xor_nonlinear_path_examples.py` |
| Figure 3; one-control-point nonlinear paths with/without endpoint scaling | `experiments.xor.joint_scale_polygonal`; `configs/experiments/xor/runners/joint_scale_polygonal.yaml` | `results/xor/xor_{2,3,5,7}h_joint_scale_{polygonal,bezier}_both_endpoints_k1/` | `tools/plotting/plot_xor_both_endpoint_scale_k1.py` |
| Figure 4 and Appendix Figure A.2 | endpoint training plus `dense_linear_stage/final_{vgg11,vgg13,vgg16,vgg19,fashion_mnist}` | `results/final_alignment_*/` | `tools/plotting/plot_larger_network_barriers.py` |
| Appendix Table A.1 | candidate grids in the five `final_*.yaml` configs | frozen selections in each final result root | no generated table; the LaTeX table summarizes the configs |
| Appendix Tables A.2 and A.3 | same runs as Figures 4/A.2 | `report_full_train/aggregates.json` and `pairs/*/*/full_profiles.json` | larger-network plotter emits the JSON values used by the tables |

Analytic equations and the method description in Sections 2–3 have no separate generated artifacts. The remaining manuscript figures are the appendix profiles already mapped above.

## 1. Controlled XOR endpoints

The canonical full-batch width-2 run uses the eight retained successful seeds:

```bash
poetry run python -m experiments.xor.basin_test \
  --output results/xor/xor_2h_15seeds

poetry run python tools/plotting/plot_xor_identity_swap_examples.py \
  --output-dir 'weekly_thesis_update(4)/paper_aistats2027/figures/identity_swap'
```

The robustness run uses true minibatches of two XOR examples and reshuffles every epoch:

```bash
poetry run python -m experiments.xor.train_linear_barriers \
  --hidden-size 2 --seeds 2,4,5,9,10,11,12,14 \
  --train-batch-size 2 \
  --output results/xor_minibatch_sgd/xor_2h_trained_linear_pairs
```

Both runners train for at most 5,000 epochs with the learning-rate schedule and stopping rule described in Appendix A.1.

## 2. XOR permutation and scale alignment

For each width, first train endpoints and exhaustively evaluate all hidden-unit permutations, then run the scale-aware comparisons:

```bash
for h in 3 5 7; do
  poetry run python -m experiments.xor.train_linear_barriers \
    --hidden-size "$h" \
    --output "results/xor/xor_${h}h_trained_linear_pairs"

  poetry run python -m experiments.xor.permutation_scale \
    --hidden-size "$h" \
    --checkpoints-dir "results/xor/xor_${h}h_trained_linear_pairs/checkpoints" \
    --output "results/xor/xor_${h}h_perm_vs_scale"
done

poetry run python tools/plotting/plot_xor_width_profiles.py
```

The bounded fallback grids are frozen in `configs/experiments/xor/search/`. Final barriers use 61 interpolation points. The output JSON records pair-level values and selected settings; use those records rather than copying values from console logs.

## 3. Nonlinear XOR paths

The fixed-endpoint path runner supports quadratic Bézier and polygonal paths. For example, the width-2 representative pair in Figure 2 is seed pair `(10,14)` with one trainable interior control (`--bezier-num-bends 3`):

```bash
poetry run python -m experiments.xor.curve_fitting \
  --hidden-neurons 2 --seeds 10,14 --pairs 10-14 \
  --bezier-num-bends 3 --curve-objective mean_loss \
  --output results/xor/xor_2h_pair10-14_figure2
```

The two uncropped representative decision-boundary strips used in the
submitted Figure 2 are retained beside the paper because the original runs did
not record a machine-readable crop/rename step. The compositor deterministically
adds labels and assembles those retained panels. This is the only paper figure
whose final visual assembly starts from retained panels rather than directly
from result JSON/NPZ files.

The paper's one-interior-point endpoint-scaling sweep is fully specified by the Slurm wrapper. It runs both path families, widths 2/3/5/7, no permutation, 1,500 Adam steps, learning rate 0.05, scale penalty `1e-4`, 31 fitting samples, and 501 final evaluation points:

```bash
for h in 2 3 5 7; do
  for family in polygonal bezier; do
    sbatch --export=ALL,HIDDEN_SIZE="$h",CURVE_TYPE="$family",SCALE_ENDPOINTS=both,INTERNAL_PARAMETERIZATION=absolute,INTERNAL_POINTS=1,EXPERIMENT_TAG=both_endpoints_k1,EVAL_POINTS=501 \
      ops/slurm/xor/run_joint_scale_polygonal.sh
  done
done

poetry run python tools/plotting/plot_xor_both_endpoint_scale_k1.py
poetry run python tools/plotting/compose_xor_nonlinear_path_examples.py
```

## 4. Larger-network validation

### 4.1 Train six independent endpoints

For CIFAR-10, generate the six seed runs expected by the final configs. Run each command first with `--dry-run`, then without it:

```bash
for model in VGG11 VGG13 VGG16 VGG19; do
  lower=$(printf '%s' "$model" | tr '[:upper:]' '[:lower:]')
  bash ops/slurm/training_stage/submit_endpoints.sh \
    "output_root=results/training_stage_${lower}_final" \
    "model=${model}"
done
```

For Fashion-MNIST:

```bash
bash ops/slurm/fashion_mnist/submit_endpoints.sh
```

The frozen splits, data normalization, optimizer, checkpoint schedule, and six seeds are in `configs/experiments/training_stage/default.yaml` and `configs/experiments/fashion_mnist/default.yaml`.

### 4.2 Run final-endpoint alignment

Each config evaluates raw interpolation, WM, WM plus fixed-permutation scale refinement, Sinkhorn, joint Sinkhorn-scale, and fixed-Sinkhorn-permutation scale refinement. Hyperparameters are selected per endpoint pair from validation data only.

```bash
for cfg in final_vgg11 final_vgg13 final_vgg16 final_vgg19 final_fashion_mnist; do
  bash ops/slurm/dense_linear_stage/submit_all.sh \
    --config-name "dense_linear_stage/${cfg}"
done
```

After the standard reports finish, evaluate frozen alignments on the complete endpoint-training split:

```bash
bash ops/slurm/dense_linear_stage/submit_full_train.sh \
  results/final_alignment_vgg11_cifar10 \
  results/final_alignment_vgg13_cifar10 \
  results/final_alignment_vgg16_cifar10 \
  results/final_alignment_vgg19_cifar10_atol5e5 \
  results/final_alignment_fashion_mnist
```

Regenerate both bar plots and their machine-readable values:

```bash
poetry run python tools/plotting/plot_larger_network_barriers.py
```

The script reads three disjoint pairs `(0,1)`, `(2,3)`, and `(4,5)`, checks for finite values, computes sample standard deviation (`ddof=1`), and writes result-local outputs and manuscript figures/data.

## Validation and expected limitations

```bash
PYTHONPATH=.:src poetry run pytest -q
find ops -name '*.sh' -print0 | xargs -0 -n1 bash -n
```

The XOR checks run locally. Full larger-network reproduction is intentionally not a lightweight test: it downloads public datasets, trains 30 networks, performs pair-specific alignment grids, and requires GPU cluster time. Existing results are retained for audit, but the documented pipeline regenerates endpoints and alignments from scratch.
