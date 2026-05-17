# Thesis Manifest

This manifest maps the experiments and figures referenced by `draft.tex` to the scripts, configs, result trees, and thesis-facing figure aliases preserved in this repository.

## Repository Notes

- Main thesis source: `thesis/tex/draft.tex`
- Root compatibility wrapper: `draft.tex`
- Figure alias directory: `thesis/figures/aliases/`
- Figure provenance index: `thesis/figures/source_index/figure_sources.json`
- Canonical code package: `src/mode_connectivity/`
- Experiment runner package: `experiments/`
- Utility runner package: `tools/`
- Missing `references.bib` remains a known repository gap.

## Experiment Map

### Curved-Path VGG16 Experiments

- Thesis sections:
  `Mode Connectivity via Curved Paths`
- Primary scripts:
  `experiments.curves.garipov_endpoints`,
  `experiments.curves.garipov_curve`,
  `experiments.curves.evaluate_garipov_curve`,
  `experiments.curves.evaluate_paths`,
  `tools.plotting.plot_connectivity`
- Slurm launchers:
  `ops/slurm/endpoints/train_endpoints.sh`,
  `ops/slurm/endpoints/transform_endpoint.sh`,
  `ops/slurm/curves/run_bezier_experiment.sh`,
  `ops/slurm/curves/evaluate_curve_checkpoint.sh`
- Primary configs:
  `configs/experiments/curves/runs/*.yaml`,
  `configs/experiments/curves/{_base,pairs,geometry}/*.yaml`
- Preserved result roots:
  `results/vgg16/cifar10/curves/standard*`,
  `results/vgg16/cifar10/advanced_geometry*`,
  `plots/connectivity_*`
- Figure aliases:
  `combined_connectivity_te_only.png`,
  `randpermute_te_only.png`

### Hyperplane-Constrained Path Finding

- Thesis section:
  `Hyperplane-Constrained Path Finding`
- Primary scripts:
  `experiments.curves.garipov_polygon`,
  `experiments.curves.random_plane`,
  `experiments.curves.symmetry_plane`,
  `experiments.curves.evaluate_paths`
- Slurm launchers:
  `ops/slurm/curves/run_polychain_experiment.sh`,
  `ops/slurm/curves/evaluate_curve_checkpoint.sh`
- Primary configs:
  `configs/experiments/curves/runs/*.yaml`,
  `configs/experiments/curves/geometry/*.yaml`
- Preserved result roots:
  `results/vgg16/cifar10/advanced_geometry*`
- Figure aliases:
  `random_plane_midpoint_codim1_5_10_30_seed0-seed1.png`

### Controlled LMC Benchmark

- Thesis section:
  `Controlled LMC Benchmark with Known Permutations`
- Primary scripts:
  `experiments.lmc.pytorch_vgg16_lmc_connected_pair`,
  `experiments.lmc.materialize_pytorch_vgg16_independent_pair`,
  `experiments.lmc.evaluate_pytorch_vgg_pair`,
  `experiments.lmc.evaluate_pytorch_vgg_split_suite`,
  `experiments.lmc.benchmark_alignment`,
  `tools.plotting.plot_pytorch_vgg_split_suite`,
  `tools.plotting.plot_pytorch_vgg_split_suite_wm`,
  `tools.plotting.plot_pytorch_vgg_split_suite_wm_test_acc_relative_barrier`
- Slurm launchers:
  `ops/slurm/lmc/run_resume.sh`,
  `ops/slurm/lmc/run_from_scratch.sh`,
  `ops/slurm/lmc/package_independent_pair.sh`,
  `ops/slurm/lmc/evaluate_split_suite.sh`,
  `ops/slurm/lmc/benchmark_alignment.sh`,
  `ops/slurm/lmc/verify_connectivity.sh`
- Primary configs:
  `configs/experiments/lmc/runs/*.yaml`,
  `configs/experiments/lmc/splits/*.yaml`,
  `configs/experiments/lmc/analysis/benchmark_alignment.yaml`
- Preserved result roots:
  `results/vgg16/cifar10/endpoints/pytorch_vgg_lmc_connected_*`,
  `results/vgg16/cifar10/endpoints/pytorch_vgg_independent_existing`,
  `results/analysis/pytorch_vgg_split_wm/*`
- Figure aliases:
  `construction.png`,
  `test_accuracy_barrier.png`

### XOR Experiment: Validating Alignment on a Minimal Problem

- Thesis sections:
  `XOR Experiment: Validating Alignment on a Minimal Problem`
- Primary scripts:
  `experiments.xor.permutation_scale`
- Slurm launchers:
  `ops/slurm/xor/run_permutation_scale.sh`
- Primary configs:
  `configs/experiments/xor/runners/permutation_scale.yaml`,
  `configs/experiments/xor/search/permutation_scale.yaml`,
  `configs/experiments/xor/search/sinkhorn_permutation.yaml`,
  `configs/experiments/xor/search/sinkhorn_permutation_scale.yaml`
- Preserved result roots:
  `results/xor/xor_3h_perm_vs_scale`,
  `results/xor/xor_5h_perm_vs_scale`,
  `results/xor/xor_7h_perm_vs_scale`
- Figure aliases:
  `success.png`,
  `unsuccesfull_same_boundary.png`,
  `unsuccesgul_diff_boundary.png`,
  `3h_xor.png`,
  `h5_xor.png`,
  `h7_xor.png`

### XOR Experiment: Validating Mode Connectivity on a Minimal Problem

- Thesis sections:
  `XOR Experiment: Validating Mode Connectivity on a Minimal Problem`,
  `XOR Experiments`
- Primary scripts:
  `experiments.xor.basin_test`,
  `experiments.xor.train_linear_barriers`,
  `experiments.xor.curve_fitting`
- Slurm launchers:
  `ops/slurm/xor/run_train_linear_barriers.sh`
- Primary configs:
  `configs/experiments/xor/runners/*.yaml`,
  `configs/experiments/xor/search/*.yaml`
- Preserved result roots:
  `results/xor/xor_2h_15seeds`,
  `results/xor/xor_2h_seeds9_10_12_14_curves`,
  `results/xor/xor_3h_unified`,
  `results/xor/xor_5h_trained_linear_pairs`
- Figure aliases:
  `2-4_interpolation.png`,
  `9-12_interpolation.png`,
  `11-14_interpolation.png`,
  `2-4_successful.png`,
  `loss_alignemt.png`,
  `unsuccesful_10-14_bezier_3bend.png`,
  `3hidden_xor_bezier.png`

### Scale-Aware Alignment for Linear Mode Connectivity

- Thesis section:
  `Scale-Aware Alignment for Linear Mode Connectivity`
- Primary scripts:
  `experiments.sinkhorn.vgg_cifar_alignment_sweep`,
  `experiments.sinkhorn.vgg_cifar_three_way_comparison`,
  `experiments.sinkhorn.vgg_cifar_three_way_barriers`
- Slurm launchers:
  `ops/slurm/sinkhorn/run_alignment_sweep.sh`,
  `ops/slurm/sinkhorn/run_three_way_comparison.sh`,
  `ops/slurm/verification/check_model_functional_equivalence.sh`
- Primary configs:
  `configs/experiments/sinkhorn/runs/vgg11_cifar_perm_only.yaml`
- Preserved result roots:
  `results/vgg11/cifar10/*`,
  `results/vgg13/cifar10/*`,
  `results/vgg16/cifar10/raw_pth_align_sweep*`,
  `results/vgg19/cifar10/*`,
  `results/vgg_cifar10_three_way_barriers/*`
- Figure aliases:
  `vgg11_sin.png`,
  `vgg13_sin.png`,
  `vgg16_sin.png`,
  `vgg19_sin.png`,
  `barplot_vggs.png`

## Figure Alias Map

| Alias in `draft.tex` | Preserved source | Status |
| --- | --- | --- |
| `example_loss_landscapev2.png` | `plots/pseudo_loss_landscape_two_minima.png` | high-confidence replacement |
| `imabarrier_example_v3ge.png` | `plots/loss_barrier_thresholds_pseudo.png` | high-confidence replacement |
| `enzari.png` | `archive/legacy/weekly_updates/weekly_thesis_update/entezari_lmc_beamer/image.png` | archived source |
| `combined_connectivity_te_only.png` | `plots/connectivity_reg_comparison_with_linear_yaxis_test_error_mixed_sources.png` | high confidence |
| `randpermute_te_only.png` | `plots/connectivity_trainaug_seedseed_vs_seedrandperm_curve_only_test_error_panel.png` | high confidence |
| `canvaxd_v2.png` | placeholder in alias layer | unresolved |
| `random_plane_midpoint_codim1_5_10_30_seed0-seed1.png` | `results/vgg16/cifar10/advanced_geometry_trainaug/figures/random_plane_midpoint_codim1_5_10_30_seed0-seed1.png` | exact |
| `construction.png` | placeholder in alias layer | unresolved |
| `test_accuracy_barrier.png` | `plots/pytorch_vgg_split_suite_wm_test_acc_relative_barrier_vs_distance.png` | high confidence |
| `success.png` | placeholder in alias layer | unresolved |
| `unsuccesfull_same_boundary.png` | placeholder in alias layer | unresolved |
| `unsuccesgul_diff_boundary.png` | placeholder in alias layer | unresolved |
| `2-4_interpolation.png` | `results/xor/xor_2h_15seeds/plots/interpolation_2_4_before_after.png` | exact |
| `9-12_interpolation.png` | `results/xor/xor_2h_15seeds/plots/interpolation_9_12_before_after.png` | exact |
| `11-14_interpolation.png` | `results/xor/xor_2h_15seeds/plots/interpolation_11_14_before_after.png` | exact |
| `2-4_successful.png` | placeholder in alias layer | unresolved |
| `loss_alignemt.png` | `results/xor/xor_5h_trained_linear_pairs/plots/loss_2_4_before_after.png` | high confidence |
| `unsuccesful_10-14_bezier_3bend.png` | `results/xor/xor_2h_seeds9_10_12_14_curves/plots/curve_10_14_bezier_snapshots.png` | high confidence |
| `3hidden_xor_bezier.png` | `results/xor/xor_3h_unified/plots/curve_3_11_bezier_snapshots.png` | high confidence |
| `3h_xor.png` | `results/xor/xor_3h_perm_vs_scale/plots/aggregate_loss_curves_titled_no_legend.png` | high confidence |
| `h5_xor.png` | `results/xor/xor_5h_perm_vs_scale/plots/aggregate_loss_curves_titled_no_legend.png` | high confidence |
| `h7_xor.png` | `results/xor/xor_7h_perm_vs_scale/plots/aggregate_loss_curves_titled_no_legend.png` | high confidence |
| `vgg11_sin.png` | `results/vgg11/cifar10/interpolation_comparison_three_way/compare_test_loss.png` | high confidence |
| `vgg13_sin.png` | `results/vgg13/cifar10/interpolation_comparison_three_way/compare_test_loss.png` | high confidence |
| `vgg16_sin.png` | `results/vgg16/cifar10/interpolation_comparison_three_way/compare_test_loss.png` | high confidence |
| `vgg19_sin.png` | `results/vgg19/cifar10/interpolation_comparison_three_way/compare_test_loss.png` | high confidence |
| `barplot_vggs.png` | `results/vgg_cifar10_three_way_barriers/vgg_cifar10_three_way_test_loss_barriers.png` | exact |

## Root-Level Thesis-Critical Checkpoints

These remain in place because active configs and Slurm scripts reference them directly:

- `VGG11_cifar10_0.911.pth`
- `VGG11_cifar10_0.9137.pth`
- `VGG11_cifar10_0.9139.pth`
- `VGG19_cifar10_0.9051_exp_9.pth`
- `VGG19_cifar10_0.9067_exp_12.pth`
- `VGG19_cifar10_0.9071_exp_1.pth`

## Archived-but-Preserved Paths

- `archive/legacy/weekly_updates/weekly_thesis_update/`
- `archive/legacy/runtime_outputs/`
- `archive/legacy/results/tmp/`
- `archive/legacy/results/vgg16_cifar10_archive/`
- `archive/legacy/root_files/`
