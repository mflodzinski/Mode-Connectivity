# Fashion-MNIST endpoint runner

`run.py` implements dataset preparation and independent endpoint training for the paper's 10-layer, width-512 MLP. `submit_endpoints.py` creates the corresponding Slurm jobs. The resulting checkpoints are consumed by `dense_linear_stage/final_fashion_mnist.yaml`; despite its historical name, that package now evaluates final endpoints only. See [../../REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).
