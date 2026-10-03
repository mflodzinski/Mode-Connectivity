# CIFAR-10 VGG endpoint training

The paper uses this family only to train six independent endpoints for each of VGG11, VGG13, VGG16, and VGG19. Inspect each submission with `--dry-run` first:

```bash
bash ops/slurm/training_stage/submit_endpoints.sh --dry-run \
  output_root=results/training_stage_vgg11_final model=VGG11
```

Omit `--dry-run` to submit. Repeat for VGG13/16/19 using the output roots named in [../../../REPRODUCIBILITY.md](../../../REPRODUCIBILITY.md). The preparation job downloads CIFAR-10 and freezes split indices and a protocol hash; the six GPU tasks then train seeds 0–5. `run_cpu.sh` and `run_gpu.sh` are scheduler workers, not user-facing entry points.
