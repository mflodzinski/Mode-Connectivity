# Fashion-MNIST endpoint training

The final paper benchmark needs six independently initialized 10-layer, width-512 MLP endpoints:

```bash
bash ops/slurm/fashion_mnist/submit_endpoints.sh --dry-run
bash ops/slurm/fashion_mnist/submit_endpoints.sh
```

This writes `results/fashion_mnist_mlp10x512`, which is consumed by `configs/experiments/dense_linear_stage/final_fashion_mnist.yaml`. Scientific settings are frozen in `configs/experiments/fashion_mnist/default.yaml`.

Launchers use the repository's Poetry `.venv` by default. Set `VENV_ACTIVATE` to use a different environment.
